# HMD SDK Sender

This folder contains a Unity sender template for Pico, Quest, and generic Unity
XR/OpenXR headset projects.

The PC Python process cannot directly read Pico or Quest standalone headset SDK
poses. Read the HMD pose inside the headset Unity app, then send it to the
Dashboard over UDP. The Dashboard provider names are:

- `auto_sdk`: use OpenVR first when SteamVR is available, otherwise listen for SDK UDP.
- `sdk_udp`: listen for a Pico, Quest, or generic Unity XR/OpenXR sender.

Older names `pico_sdk`, `quest_sdk`, and `unity_xr_sdk` are accepted by the
backend as compatibility aliases, but the Dashboard now shows one merged SDK UDP
mode.

## Usage

1. Copy `Unity/HMDSdkPoseSender.cs` into a Unity XR project running on the headset.
2. Attach it to any active GameObject.
3. Set `Target Host` to the PC IP address running the Dashboard.
4. Keep `Target Port` aligned with `hmd.sdk_udp_port` in `config/config.json`
   (default `39542`).
5. Build and run the Unity app on Pico/Quest.
6. In the Dashboard, start `sdk_udp` or `auto_sdk`.

The sender uses Unity coordinates: X right, Y up, Z forward. Quaternion JSON is
sent as `[w, x, y, z]` so the PC provider can use it without guessing.

Height correction is intentionally stricter than pose tracking. The sender only
adds `actual_height` after Unity confirms a Floor tracking origin. Device/unknown
origins still provide position and rotation, but the PC will not interpret their
Y value as height above the real floor. Legacy binary packets have no origin
metadata and therefore never drive crouch/seated height correction.

If XR device pose reads fail, `Camera.main` is used only as a continuity
fallback. Camera world coordinates are sent with `tracking_origin=unknown` and
without `actual_height`, because scene and `XROrigin` parent offsets make that
Y value unsafe as a real floor height.

## Packet Shape

```json
{
  "source": "unity_xr_sdk",
  "position": [0.0, 1.65, 0.0],
  "quaternion": [1.0, 0.0, 0.0, 0.0],
  "tracking_origin": "floor",
  "actual_height": 1.65,
  "valid": true,
  "timestamp": 1783875000.0,
  "sequence": 42
}
```

When the Unity XR device pose is unavailable, the script falls back to
`Camera.main.transform`. If neither source is available, it sends `valid:false`.
