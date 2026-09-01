"""
Headless daytime CARLA recorder for a given town (for v17 testing).
Based on record_night_auto.py but Sunny weather, output -> datasets/recorded_<Town>/.
Saves rgb/<frame>.png (BGR[:3]) and semantic/<frame>.png, 1024x512, autopilot.

Usage: conda run -n carla_env python3 record_town_auto.py --town Town05 --frames 600
"""
from config import DATA
import argparse, carla, queue, numpy as np, random, sys
from pathlib import Path
from PIL import Image

SUNNY = carla.WeatherParameters(cloudiness=10, precipitation=0, sun_altitude_angle=70, wetness=0, fog_density=0)
# Night: sun below the horizon. CARLA's light manager switches street lamps on automatically
# once sun_altitude_angle < 0, and vehicle lights are enabled explicitly below.
NIGHT = carla.WeatherParameters(cloudiness=10, precipitation=0, sun_altitude_angle=-25, wetness=0, fog_density=3)
# Rain: overcast + active precipitation + standing water. precipitation_deposits drives the
# road puddles that make wet scenes read as wet; wetness drives the surface specularity.
RAIN = carla.WeatherParameters(cloudiness=80, precipitation=60, precipitation_deposits=60,
                               wetness=80, sun_altitude_angle=45, fog_density=8)
WEATHERS = {'sunny': SUNNY, 'night': NIGHT, 'rain': RAIN}

def spawn_npcs(world, n=20, anchor=None):
    """Spawn n NPC vehicles, clustered near `anchor` (the ego spawn) when given.

    A plain random.shuffle over every spawn point scatters NPCs across the WHOLE map, so the
    ego almost never meets them: Town05 with 150 NPCs over its 302 map-wide spawn points
    yielded 1.0 visible vehicle per frame, covering 1.2% of the image. Raising the count does
    not help much -- the cars are simply somewhere else. Sorting spawn points by distance to
    the ego packs the traffic into the streets the ego actually drives.

    They still disperse over time under autopilot, which is fine and realistic; what matters
    is that the run STARTS dense instead of starting empty.
    """
    bp_lib = world.get_blueprint_library()
    sps = world.get_map().get_spawn_points()
    if anchor is not None:
        sps = sorted(sps, key=lambda sp: sp.location.distance(anchor))
    else:
        random.shuffle(sps)
    npcs = []
    for sp in sps[:n]:
        a = world.try_spawn_actor(random.choice(bp_lib.filter('vehicle.*')), sp)
        if a: a.set_autopilot(True, 8000); npcs.append(a)
    return npcs

def recycle_npcs(world, npcs, spawns, ego_loc, far=140.0, lo=30.0, hi=120.0, gap=8.0):
    """Teleport NPCs that have wandered off back into the ego's neighbourhood.

    Clustering at spawn only makes the FIRST few seconds dense. Under autopilot the traffic
    disperses across the map, and by mid-clip the ego is driving alone again -- which is how
    Town05 ended up at 1.0 visible vehicle per frame despite 150 NPCs. Recycling sustains the
    density for the whole take, which is what an object-detection benchmark actually needs.

    Only vehicles beyond `far` are moved, and only into a slot at least `gap` from every other
    vehicle and no nearer than `lo` to the ego -- so nothing pops into shot or lands on top of
    another car.
    """
    cand = [sp for sp in spawns if lo <= sp.location.distance(ego_loc) <= hi]
    if not cand:
        return 0
    occupied = []
    for n in npcs:
        try:
            occupied.append(n.get_transform().location)
        except Exception:
            pass
    moved = 0
    for n in npcs:
        try:
            if n.get_transform().location.distance(ego_loc) <= far:
                continue
        except Exception:
            continue
        random.shuffle(cand)
        for sp in cand:
            if all(sp.location.distance(o) > gap for o in occupied):
                try:
                    n.set_transform(sp)
                    n.set_target_velocity(carla.Vector3D(0, 0, 0))
                    occupied.append(sp.location); moved += 1
                except Exception:
                    pass
                break
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--npcs", type=int, default=20)
    ap.add_argument("--outname", default=None, help="output dir recorded_<outname> (default=town); use to record a 2nd route of the same map")
    ap.add_argument("--spawn_mode", choices=["buildings", "highway"], default="buildings", help="buildings=dense city spawn (default); highway=most driving lanes, fewest buildings (Town04 ring)")
    ap.add_argument("--spawn_rank", type=int, default=0, help="which building-dense spawn to start from (0=densest; higher=a different part of the city)")
    ap.add_argument("--width", type=int, default=1024, help="camera width (2048 for high-res far-building detail)")
    ap.add_argument("--height", type=int, default=512, help="camera height")
    ap.add_argument("--motion_blur", type=float, default=0.0, help="motion blur intensity (0=off; Epic defaults to 0.45 which smears -96% sharpness)")
    ap.add_argument("--exposure_comp", type=float, default=0.6, help="exposure compensation to counter Epic auto-exposure darkening")
    ap.add_argument("--weather", choices=list(WEATHERS), default="sunny", help="sunny or night")
    ap.add_argument("--instance", action="store_true", help="also record the instance-segmentation camera into instance/")
    # Downstream consumers specify a lens. Vision Pilot wants 52-55 deg HFOV; our long-standing
    # default of 90 is far wider, and a wide lens compresses distance non-linearly, so lane
    # curvature and every distance-derived output are wrong however good the pixels are.
    ap.add_argument("--fov", type=float, default=90.0, help="camera horizontal FOV in degrees (90=legacy default; 55 matches Vision Pilot)")
    # CARLA pitch is POSITIVE = UP. The long-standing +15 points the camera 15 deg above the
    # horizon; at fov 90 the lens is wide enough that road still fills the lower frame, so it
    # was never noticed. Narrow to 55 around the same axis and the shot is buildings and sky
    # with almost no road. For a 55 deg lens (29.2 deg vertical at 2:1), pitch -5 puts the
    # horizon ~1/3 down and fills the lower two thirds with road -- a proper dashcam view.
    ap.add_argument("--pitch", type=float, default=15.0, help="camera pitch, POSITIVE=UP (15=legacy; use -5 with --fov 55)")
    ap.add_argument("--min_move", type=float, default=0.06, help="metres the car must travel for a frame to be saved (skips frames where traffic has it stopped)")
    ap.add_argument("--recycle_every", type=int, default=120, help="ticks between pulling stray NPCs back near the ego (0=off)")
    ap.add_argument("--recycle_dist", type=float, default=140.0, help="an NPC further than this from the ego gets recycled")
    ap.add_argument("--no_teleport", action="store_true",
                    help="never teleport on stuck; abort the take instead (exit 3) so a wrapper "
                         "can retry from a different spawn. A mid-clip teleport is a hard cut to "
                         "another part of the map -- physics stays level, but the video jumps, "
                         "which breaks Vision Pilot tracking and reads as the camera flying.")
    ap.add_argument("--settle_ticks", type=int, default=60, help="max ticks to wait for the car to settle after a teleport")
    ap.add_argument("--max_speed", type=float, default=30.0, help="m/s above which a frame is a physics artefact and is dropped")
    ap.add_argument("--stuck_limit", type=int, default=150, help="ticks stationary before teleporting to a fresh spawn (150 = 5s)")
    a = ap.parse_args()
    out = Path(DATA) / f"recorded_{a.outname or a.town}"
    (out/"rgb").mkdir(parents=True, exist_ok=True); (out/"semantic").mkdir(parents=True, exist_ok=True)

    client = carla.Client("localhost", 2000); client.set_timeout(180.0)  # mega-maps (Town12/13) tick slowly
    print(f"Loading {a.town} ..."); world = client.load_world(a.town)
    s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 1/30.0
    world.apply_settings(s); world.set_weather(WEATHERS[a.weather])
    tm = client.get_trafficmanager(8000); tm.set_synchronous_mode(True); tm.set_global_distance_to_leading_vehicle(2.0)

    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    # Default picks a BUILDING-DENSE spawn so we drive through the city rather than the
    # highway -- right for texture work, exactly wrong for Town04, whose value is its highway
    # ring (multi-lane, cars ahead in lane: the natural ACC/FCW/lane-keeping scenario).
    # --spawn_mode highway inverts the sort and prefers multi-lane road, so the ego starts on
    # the ring instead of in Town04's small village.
    try:
        blds = world.get_environment_objects(carla.CityObjectLabel.Buildings)
        blocs = [b.transform.location for b in blds]
        near = lambda sp: sum(1 for bl in blocs if sp.location.distance(bl) < 60)
        if a.spawn_mode == 'highway':
            amap = world.get_map()
            def lanes(sp):
                # Walk left then right counting driving lanes. MUST be bounded and must track
                # visited lane ids: on a DIVIDED carriageway (Town04's ring) get_left_lane()
                # crosses the median into an opposite-direction lane whose own left lane is
                # the one we came from, so a naive while-loop ping-pongs forever. That hung
                # Town04 at 99.9% CPU for two hours with zero frames recorded.
                try:
                    wp = amap.get_waypoint(sp.location)
                    if wp is None:
                        return 1
                    n = 1
                    for step in (lambda x: x.get_left_lane(), lambda x: x.get_right_lane()):
                        l = wp; seen = {(wp.road_id, wp.lane_id)}
                        for _ in range(8):                     # no real road has 8 lanes a side
                            l = step(l)
                            if l is None or l.lane_type != carla.LaneType.Driving:
                                break
                            key = (l.road_id, l.lane_id)
                            if key in seen:                    # walked back onto a lane we had
                                break
                            seen.add(key); n += 1
                    return n
                except Exception:
                    return 1
            spawns = sorted(spawns, key=lambda sp: (-lanes(sp), near(sp)))
            print(f"highway mode: top spawn has {lanes(spawns[0])} driving lanes, {near(spawns[0])} buildings within 60m")
        else:
            spawns = sorted(spawns, key=lambda sp: -near(sp))
            print(f"{len(blocs)} buildings; top spawn has {near(spawns[0])} within 60m")
    except Exception as e:
        print("building-spawn selection failed, using default:", e)
    sp_idx = min(a.spawn_rank, len(spawns) - 1)
    print(f"starting from spawn rank {sp_idx} of {len(spawns)}")
    veh = world.spawn_actor(bp.filter("vehicle.tesla.model3")[0], spawns[sp_idx])
    veh.set_autopilot(True, 8000)
    # keep the ego moving through the city (don't idle at red lights)
    tm.ignore_lights_percentage(veh, 100.0)
    tm.ignore_signs_percentage(veh, 100.0)
    tm.vehicle_percentage_speed_difference(veh, -15.0)
    npcs = spawn_npcs(world, n=a.npcs, anchor=spawns[sp_idx].location)
    print(f"Spawned {len(npcs)} NPCs clustered around the ego spawn")
    if a.weather == 'night':
        # Headlights/brake lights are the main light sources in a night street scene; without
        # them every vehicle is an unlit silhouette and the render has nothing to key on.
        ls = carla.VehicleLightState(carla.VehicleLightState.LowBeam |
                                     carla.VehicleLightState.Position |
                                     carla.VehicleLightState.Brake)
        for v in [veh] + npcs:
            try: v.set_light_state(ls)
            except Exception: pass
        try:
            world.set_weather(WEATHERS['night'])   # re-assert after actors spawn
        except Exception: pass

    cam_tf = carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=a.pitch))
    rgb_q, sem_q = queue.Queue(), queue.Queue()
    W, H = str(a.width), str(a.height)
    rgb_bp = bp.find("sensor.camera.rgb"); rgb_bp.set_attribute("image_size_x",W); rgb_bp.set_attribute("image_size_y",H); rgb_bp.set_attribute("fov",str(a.fov))
    # Epic quality enables motion blur (smears every frame -96% sharpness) -> disable it; keep TAA (fence fix).
    for attr, val in [("motion_blur_intensity", str(a.motion_blur)),
                      ("motion_blur_max_distortion", "0.0"),
                      ("motion_blur_min_object_screen_size", "0.0"),
                      ("exposure_compensation", str(a.exposure_comp))]:
        if rgb_bp.has_attribute(attr):
            rgb_bp.set_attribute(attr, val)
    sem_bp = bp.find("sensor.camera.semantic_segmentation"); sem_bp.set_attribute("image_size_x",W); sem_bp.set_attribute("image_size_y",H)
    sem_bp.set_attribute("fov", str(a.fov))   # must match the RGB camera or labels misalign
    rgb_cam = world.spawn_actor(rgb_bp, cam_tf, attach_to=veh); sem_cam = world.spawn_actor(sem_bp, cam_tf, attach_to=veh)
    rgb_cam.listen(rgb_q.put); sem_cam.listen(sem_q.put)

    # Instance segmentation: semantic cannot separate two touching cars -- they share a label,
    # so the generator paints the pair as one region. That is the "some parts painted red some
    # isn't" artefact. This camera gives every actor its own id, and the boundaries between ids
    # are merged into the edge channel at training time.
    inst_cam = None; inst_q = None
    if a.instance:
        (out/"instance").mkdir(parents=True, exist_ok=True)
        inst_q = queue.Queue()
        inst_bp = bp.find("sensor.camera.instance_segmentation")
        inst_bp.set_attribute("image_size_x", W); inst_bp.set_attribute("image_size_y", H)
        inst_bp.set_attribute("fov", str(a.fov))
        inst_cam = world.spawn_actor(inst_bp, cam_tf, attach_to=veh)
        inst_cam.listen(inst_q.put)

    for _ in range(30):  # warm-up
        world.tick(); rgb_q.get(timeout=5.0); sem_q.get(timeout=5.0)
        if inst_q is not None: inst_q.get(timeout=5.0)
    print(f"Recording {a.frames} frames ...")
    # The ego already ignores lights/signs, but NPC traffic can still box it in. Frames where
    # nothing moves are worthless (identical pixels) and they poison the flicker metrics too,
    # so don't save them: keep ticking until the car has actually travelled, and if it stays
    # boxed in past --stuck_limit ticks, teleport to a fresh spawn rather than burn the take.
    i = 0; stuck = 0; ticks = 0; recycled = 0; bad = 0; last = veh.get_transform().location
    spare = list(spawns); random.shuffle(spare)
    # Per-frame ego speed, one line per SAVED frame. Downstream planners (Vision Pilot's
    # record.sh) otherwise assume a constant 20 m/s, which makes every longitudinal result --
    # braking, following distance, ACC -- meaningless regardless of how good the imagery is.
    # It must be written per saved frame, not per tick: frames below --min_move are dropped,
    # so a tick-indexed log would desynchronise from the images.
    speeds = []
    while i < a.frames:
        world.tick(); ticks += 1
        if a.recycle_every and npcs and ticks % a.recycle_every == 0:
            recycled += recycle_npcs(world, npcs, spawns, veh.get_transform().location,
                                     far=a.recycle_dist)
        r = rgb_q.get(timeout=5.0); sm = sem_q.get(timeout=5.0)
        loc = veh.get_transform().location
        moved = loc.distance(last)
        if moved < a.min_move:
            stuck += 1
            if stuck >= a.stuck_limit and a.no_teleport:
                print(f"  stuck {stuck} ticks and --no_teleport set -> aborting take at frame {i}")
                sys.exit(3)
            if stuck >= a.stuck_limit and spare:
                sp = spare.pop()
                # Teleport recovery used to end the take in a crash. set_transform drops the car
                # AT the spawn transform, CARLA spawn points sit above the road, and
                # set_target_velocity does not reliably take effect in the same tick -- so the
                # car free-fell through the 20 settle ticks and tumbled on landing. Measured on
                # Town10HD night: velocity ramped 26 -> 38 m/s (85 mph) in ~1.2 s, which is
                # exactly 9.8 m/s^2, then collapsed to 17.8 and 11.9 as it hit and bounced.
                # Those frames were saved, so the clip ends with the camera cartwheeling.
                #
                # Fix: lift clear of the surface, zero BOTH linear and angular velocity and keep
                # re-zeroing while physics settles, then refuse to resume until the car is
                # actually upright and slow. No frame is saved during any of this.
                sp = carla.Transform(
                    carla.Location(sp.location.x, sp.location.y, sp.location.z + 0.30),
                    sp.rotation)
                veh.set_transform(sp)
                print(f"  stuck {stuck} ticks -> respawned at a new point")
                settled = False
                for k in range(a.settle_ticks):
                    if k < 8:      # one call before a single tick is not enough
                        veh.set_target_velocity(carla.Vector3D(0, 0, 0))
                        veh.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                    world.tick(); rgb_q.get(timeout=5.0); sem_q.get(timeout=5.0)
                    if inst_q is not None: inst_q.get(timeout=5.0)
                    if k < 12:
                        continue
                    vv = veh.get_velocity()
                    sp_ = (vv.x ** 2 + vv.y ** 2 + vv.z ** 2) ** 0.5
                    rot = veh.get_transform().rotation
                    # Do NOT require the car to be stationary. The traffic manager resumes
                    # driving it the instant it lands, so a speed gate can never be satisfied --
                    # the Town10HD redo logged "not settled after 60 ticks" at roll 0.0
                    # pitch -0.1, i.e. perfectly upright and simply moving. Orientation and a
                    # sane (non-falling) speed are what actually matter.
                    if sp_ < 12.0 and abs(rot.roll) < 8.0 and abs(rot.pitch) < 8.0:
                        settled = True
                        break
                if not settled:
                    rot = veh.get_transform().rotation
                    print(f"  WARNING: not settled after {a.settle_ticks} ticks "
                          f"(roll {rot.roll:.1f} pitch {rot.pitch:.1f})")
                stuck = 0; last = veh.get_transform().location
            continue                                    # do not save a frozen frame
        stuck = 0; last = loc
        v = veh.get_velocity()
        spd = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
        # Second line of defence: a real collision can still leave the car tumbling, and those
        # frames are worse than useless -- Vision Pilot reads frame_speed.txt as truth, so an
        # 85 mph artefact corrupts every longitudinal decision it makes.
        rot = veh.get_transform().rotation
        if spd > a.max_speed or abs(rot.roll) > 8.0 or abs(rot.pitch) > 8.0:
            bad += 1
            if bad == 1 or bad % 25 == 0:
                print(f"  dropped frame: {spd:.1f} m/s roll {rot.roll:.1f} pitch {rot.pitch:.1f} "
                      f"({bad} dropped)")
            continue
        speeds.append(spd)   # m/s
        Image.fromarray(np.array(r.raw_data).reshape((r.height,r.width,4))[:,:,:3]).save(out/"rgb"/f"{i:06d}.png")
        Image.fromarray(np.array(sm.raw_data).reshape((sm.height,sm.width,4))[:,:,:3]).save(out/"semantic"/f"{i:06d}.png")
        if inst_q is not None:
            im_ = inst_q.get(timeout=5.0)
            Image.fromarray(np.array(im_.raw_data).reshape((im_.height,im_.width,4))[:,:,:3]).save(out/"instance"/f"{i:06d}.png")
        i += 1
        if i % 200 == 0: print(f"  {i}/{a.frames}")
    (out / "frame_speed.txt").write_text("\n".join(f"{x:.4f}" for x in speeds) + "\n")
    if speeds:
        print(f"frame_speed.txt: {len(speeds)} lines, mean {sum(speeds)/len(speeds):.1f} m/s "
              f"({sum(speeds)/len(speeds)*2.237:.0f} mph), max {max(speeds):.1f} m/s")
    if a.recycle_every: print(f"recycled {recycled} stray NPCs back near the ego")
    if bad: print(f"dropped {bad} unstable frames (tumbling or >{a.max_speed} m/s)")
    print(f"Done -> {out}")
    s.synchronous_mode=False; world.apply_settings(s)
    rgb_cam.stop(); sem_cam.stop(); rgb_cam.destroy(); sem_cam.destroy()
    for n in npcs: n.destroy()
    veh.destroy()

if __name__ == "__main__":
    main()
