# VRM humanoid avatar

`anime_female.vrm.glb` is the official VRM 1.0 sample avatar
`samples/VRM1_Constraint_Twist_Sample/vrm/VRM1_Constraint_Twist_Sample.vrm` from
https://github.com/vrm-c/vrm-specification, downloaded 2026-08-29 and renamed
`.glb` because VRM *is* glTF 2.0 with extensions and every reader here treats it
as one.

Chosen over the SMPL body and over the CC0 low-poly packs because VRM specifies
its humanoid bone vocabulary (`VRMC_vrm`), so `tools/vrm_retarget.SMPL_TO_VRM` is
written once against a standard rather than against one artist's naming: 22 of
SMPL's 24 joints map one-to-one, and the two that do not are SMPL's hand joints,
which this corpus holds at identity (max |axis-angle| 4.7e-03). Any other VRM
drops in with no code change.

**Appearance only, and one stated limitation.** The avatar contributes geometry,
clothes, hair and proportions; every angle on screen is the motion the model
produced, checked by the joint gate in `render_avatar_video.skin` on the SMPL
path and by the shared stage on this one. VRM's spring-bone and constraint
extensions are **not** evaluated, so the hair and the hem are rigid -- a
swinging skirt would be motion this pipeline did not generate, and inventing it
is exactly what a visualisation must not do.
