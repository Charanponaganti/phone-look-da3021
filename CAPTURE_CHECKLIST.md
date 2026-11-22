# Capture checklist (tonight)

## Goal
12–20 scenes, each with a **normal** shot and a **Night/HDR** shot of the same view.

## Naming (important)
Put files in `data/raw/pairs/`:

```
scene01_src.jpg   ← normal / default camera
scene01_tgt.jpg   ← Night or HDR / “best” mode
scene02_src.jpg
scene02_tgt.jpg
...
```

`src` = source (ordinary look)  
`tgt` = target (the look you want to copy)

## How to shoot each scene
1. Fix the phone on a table / books / wall lean (less shake).
2. Take **src** in normal photo mode.
3. Switch to Night or HDR (whichever looks clearly different).
4. Take **tgt** from the **same spot**, same framing, within a few seconds.
5. Prefer static scenes (no walking people in the middle).

## Good scene list (mix these)
- Hostel desk under LED
- Face / selfie-ish portrait (consent if someone else)
- Window / backlight
- Whiteboard or printed text
- Corridor
- Food / colourful object
- Outdoor dusk
- Plant / leaves (texture)
- Bookshelf
- Bathroom mirror scene (careful with fog/lights)

## Optional bursts (extra marks)
For 4–5 scenes, also shoot 6–8 quick frames in normal mode and put them in:

```
data/raw/bursts/scene01/0001.jpg
data/raw/bursts/scene01/0002.jpg
...
```

## Transfer to laptop
Copy into `phone_look_project/data/raw/pairs/` then run:

```bash
cd phone_look_project
python -m src.align_pairs
```
