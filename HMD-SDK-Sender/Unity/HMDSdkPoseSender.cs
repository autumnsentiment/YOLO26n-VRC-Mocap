using System;
using System.Collections.Generic;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.XR;

public class HMDSdkPoseSender : MonoBehaviour
{
    public string targetHost = "127.0.0.1";
    public int targetPort = 39542;
    public float sendRate = 90.0f;
    public string source = "unity_xr_sdk";
    public string trackingOrigin = "unknown";
    public bool useCenterEye = true;
    public bool setFloorTrackingOrigin = true;

    private UdpClient udpClient;
    private IPEndPoint targetEndpoint;
    private float nextSendTime;
    private bool floorOriginActive;
    private long sequence;

    private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

    private void Start()
    {
        udpClient = new UdpClient();
        targetEndpoint = new IPEndPoint(IPAddress.Parse(targetHost), targetPort);

        if (setFloorTrackingOrigin)
        {
            TrySetFloorTrackingOrigin();
        }
    }

    private void OnDestroy()
    {
        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
        }
    }

    private void Update()
    {
        float interval = 1.0f / Mathf.Max(1.0f, sendRate);
        if (Time.unscaledTime < nextSendTime)
        {
            return;
        }
        nextSendTime = Time.unscaledTime + interval;

        Vector3 position;
        Quaternion rotation;
        bool floorRelativePose = TryGetXRPose(out position, out rotation);
        bool valid = floorRelativePose || TryGetCameraPose(out position, out rotation);
        SendPose(valid, floorRelativePose, position, rotation);
    }

    private bool TryGetXRPose(out Vector3 position, out Quaternion rotation)
    {
        XRNode node = useCenterEye ? XRNode.CenterEye : XRNode.Head;
        InputDevice device = InputDevices.GetDeviceAtXRNode(node);
        bool hasPosition = device.TryGetFeatureValue(CommonUsages.devicePosition, out position);
        bool hasRotation = device.TryGetFeatureValue(CommonUsages.deviceRotation, out rotation);
        return device.isValid && hasPosition && hasRotation;
    }

    private bool TryGetCameraPose(out Vector3 position, out Quaternion rotation)
    {
        Camera cam = Camera.main;
        if (cam == null)
        {
            position = Vector3.zero;
            rotation = Quaternion.identity;
            return false;
        }

        Transform t = cam.transform;
        position = t.position;
        rotation = t.rotation;
        return true;
    }

    private void TrySetFloorTrackingOrigin()
    {
        trackingOrigin = "unknown";
        floorOriginActive = false;
        List<XRInputSubsystem> subsystems = new List<XRInputSubsystem>();
        SubsystemManager.GetInstances(subsystems);

        for (int i = 0; i < subsystems.Count; i++)
        {
            XRInputSubsystem subsystem = subsystems[i];
            if (subsystem != null && subsystem.running)
            {
                if (subsystem.TrySetTrackingOriginMode(TrackingOriginModeFlags.Floor))
                {
                    trackingOrigin = "floor";
                    floorOriginActive = true;
                    return;
                }
                if (subsystem.GetTrackingOriginMode() == TrackingOriginModeFlags.Floor)
                {
                    trackingOrigin = "floor";
                    floorOriginActive = true;
                    return;
                }
                if ((subsystem.GetTrackingOriginMode() & TrackingOriginModeFlags.Device) != 0)
                {
                    trackingOrigin = "device";
                }
            }
        }
    }

    private void SendPose(
        bool valid,
        bool floorRelativePose,
        Vector3 position,
        Quaternion rotation
    )
    {
        if (udpClient == null || targetEndpoint == null)
        {
            return;
        }

        string json;
        double timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
        if (valid)
        {
            string origin = floorRelativePose ? GetCurrentTrackingOrigin() : "unknown";
            string heightField = floorRelativePose && IsFloorLikeOrigin(origin)
                ? string.Format(Invariant, ",\"actual_height\":{0}", Mathf.Max(0.0f, position.y))
                : string.Empty;
            json = string.Format(
                Invariant,
                "{{\"source\":\"{0}\",\"position\":[{1},{2},{3}],\"quaternion\":[{4},{5},{6},{7}],\"tracking_origin\":\"{8}\"{9},\"valid\":true,\"timestamp\":{10},\"sequence\":{11}}}",
                EscapeJson(source),
                position.x, position.y, position.z,
                rotation.w, rotation.x, rotation.y, rotation.z,
                EscapeJson(origin),
                heightField,
                timestamp,
                sequence
            );
        }
        else
        {
            json = string.Format(
                Invariant,
                "{{\"source\":\"{0}\",\"tracking_origin\":\"{1}\",\"valid\":false,\"timestamp\":{2},\"sequence\":{3}}}",
                EscapeJson(source),
                EscapeJson(GetCurrentTrackingOrigin()),
                timestamp,
                sequence
            );
        }

        byte[] bytes = Encoding.UTF8.GetBytes(json);
        udpClient.Send(bytes, bytes.Length, targetEndpoint);
        sequence++;
    }

    private static string EscapeJson(string value)
    {
        return (value ?? string.Empty).Replace("\\", "\\\\").Replace("\"", "\\\"");
    }

    private string GetCurrentTrackingOrigin()
    {
        List<XRInputSubsystem> subsystems = new List<XRInputSubsystem>();
        SubsystemManager.GetInstances(subsystems);
        for (int i = 0; i < subsystems.Count; i++)
        {
            XRInputSubsystem subsystem = subsystems[i];
            if (subsystem == null || !subsystem.running)
            {
                continue;
            }

            TrackingOriginModeFlags mode = subsystem.GetTrackingOriginMode();
            if ((mode & TrackingOriginModeFlags.Floor) != 0)
            {
                trackingOrigin = "floor";
                floorOriginActive = true;
                return trackingOrigin;
            }
            if ((mode & TrackingOriginModeFlags.Device) != 0)
            {
                trackingOrigin = "device";
                floorOriginActive = false;
                return trackingOrigin;
            }
        }
        floorOriginActive = false;
        trackingOrigin = "unknown";
        return trackingOrigin;
    }

    private static bool IsFloorLikeOrigin(string origin)
    {
        string value = (origin ?? string.Empty).ToLowerInvariant();
        return value == "floor" || value == "stage" || value == "standing" || value == "local_floor";
    }
}
