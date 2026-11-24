# Capture checklist (tonight)

## Your phone
**iQOO Neo 9 Pro** → source device for personal pairs.

| Role | How on this phone |
|------|-------------------|
| **src** | Stock **Photo** mode, default / auto |
| **tgt** | **Night** *or* **HDR** (pick the one that looks clearly better) |

Goal for your pairs: teach the model *your* phone’s everyday look → its “expensive ISP” look, on top of DPED iPhone↔Canon learning.

## Goal
12–20 scenes, each with a **normal** shot and a **Night/HDR** shot of the same view.

## Naming (important)
Put files in `data/raw/pairs/`:

```
scene01_src.jpg   ← Photo mode (default)
scene01_tgt.jpg   ← Night or HDR
scene02_src.jpg
scene02_tgt.jpg
...
```

`src` = ordinary Neo 9 Pro look  
`tgt` = the look you want to copy

## How to shoot each scene (iQOO Neo 9 Pro)
1. Prop the phone (table / books / wall lean) — less shake.
2. **src:** Photo mode → tap to focus → shoot.
3. Switch to **Night** (dim scenes) or **HDR** (daylight / window / mixed light).
4. **tgt:** same spot, same framing, within a few seconds.
5. Prefer static scenes (no people walking through).
6. Avoid ultra-wide / zoom mismatches between src and tgt — same lens.

## Scene checklist (tick as you go)
Mix indoor / outdoor / faces / text / colour.

| # | Scene idea | Why it helps | src | tgt |
|---|------------|--------------|-----|-----|
| 01 | Hostel desk under LED | Warm cast, noise | ☐ | ☐ |
| 02 | Face / portrait (consent) | Skin tones | ☐ | ☐ |
| 03 | Window / backlight | Highlights & shadows | ☐ | ☐ |
| 04 | Whiteboard / printed text | Sharpness / readability | ☐ | ☐ |
| 05 | Corridor | Mixed lighting | ☐ | ☐ |
| 06 | Food / colourful object | Saturation | ☐ | ☐ |
| 07 | Outdoor dusk | Low light | ☐ | ☐ |
| 08 | Plant / leaves | Fine texture | ☐ | ☐ |
| 09 | Bookshelf | Structure & colour | ☐ | ☐ |
| 10 | Notice board / poster | Text + colour | ☐ | ☐ |
| 11 | Bathroom / mirror lights | Hard light | ☐ | ☐ |
| 12 | Stairwell | Dim indoor | ☐ | ☐ |
| 13 | Outdoor building facade | Daylight baseline | ☐ | ☐ |
| 14 | Desk lamp close-up | Strong local light | ☐ | ☐ |
| 15 | Classroom / lab bench | Everyday usefulness | ☐ | ☐ |
| 16 | Your choice | Fill gaps | ☐ | ☐ |

## Optional bursts (extra marks)
For 4–5 scenes, also shoot 6–8 quick frames in Photo mode:

```
data/raw/bursts/scene01/0001.jpg
data/raw/bursts/scene01/0002.jpg
...
```

## Transfer to laptop
Copy into `phone_look_project/data/raw/pairs/` (keep names `scene##_src/tgt`) then tell me — I’ll fold them into align + retrain with the DPED subset.
