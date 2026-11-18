"""Builds the portfolio site.

How to use:
    1. Edit the content in the PROFILE, PROJECTS, SIDE_QUESTS, CAREER, SKILLS and TROPHIES
       sections below. Everything on the site comes from here.
    2. Run:  python build.py
    3. Commit and push. GitHub Pages serves the HTML files this script writes.

It only uses the Python standard library, so there is nothing to install.
"""

import html
import json
from pathlib import Path

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------------------
# PROFILE: who you are. Fill in EMAIL when you're ready; it stays hidden while empty.
# ---------------------------------------------------------------------------------------
PROFILE = {
    "name": "Thirunavukarasu Palanisamy",
    "other_names": ["Thirunavukarasu P", "Thiru"],
    "short_name": "Thiru",
    "title": "Robotics Engineer",
    "headline": "Robotics engineer working on robot perception, robot learning and control",
    "base_url": "https://thiru51.github.io/",
    "location": "Chennai, India",
    "school": "Indian Institute of Technology Madras",
    "degree": "B.Tech, Naval Architecture and Ocean Engineering, IIT Madras (2026)",
    "email": "",  # e.g. "you@example.com"
    "github": "https://github.com/thiru51",
    "linkedin": "https://www.linkedin.com/in/thirunavukarasu-p-65262827a/",
    "intro": (
        "I build the software that lets robots see, move and decide what to do next: "
        "perception models, learned locomotion and driving policies, state estimation and "
        "control. I write my own implementations, test them properly, and report what "
        "actually worked, including the parts that didn't."
    ),
    "open_to": "Robotics, perception, robot learning and autonomy roles",
}

# ---------------------------------------------------------------------------------------
# PROJECTS: the "levels". Each one gets its own page at projects/<slug>.html
# ---------------------------------------------------------------------------------------
PROJECTS = [
    {
        "slug": "pedestrian-crossing-intent",
        "title": "Will that pedestrian cross?",
        "subtitle": "Pedestrian crossing-intent prediction from pose, gesture and scene context",
        "short": "Predicts whether a pedestrian will step onto the road 1.5 seconds before they do, using body pose and the scene around them.",
        "score": "0.633 AP on real video (pose alone: 0.385)",
        "tags": ["Perception", "Transformers", "CARLA"],
        "repo": "pedestrian-intent-forecast",
        "status": "Complete on JAAD real-world video. PIE evaluation is next.",
        "problem": [
            "Most pedestrian predictors watch where a person has walked and extrapolate. By the time the path gives it away, the car has already lost reaction time.",
            "Body language shows intent earlier: turning the head, leaning, a hand gesture. This project uses that to predict a crossing before the first step.",
        ],
        "built": [
            "Parsers for the JAAD (346 videos) and PIE driving datasets, checked against the real released files.",
            "A CARLA simulator collector that records crossing scenarios with exact labels (222 episodes).",
            "Pose features from MediaPipe, a scene-context encoder and a gesture taxonomy modelled on TITAN.",
            "A small temporal transformer over 16-frame windows with an intent head and a trajectory head.",
            "An ONNX export with a numerical round-trip check, and 59 unit tests.",
        ],
        "results": [
            "On real JAAD video (132,700 pedestrian-frames, 3 seeds, split strictly by video): <b>0.633 AP</b> with pose plus scene context, against <b>0.385 AP</b> for pose alone. The per-seed ranges do not overlap.",
            "In CARLA simulation: 0.943 AP fused against 0.830 pose-only.",
            "CARLA suggested a 2.5x bigger gain on occluded pedestrians. That did not reproduce on real video (0.8x), so I report it as a simulation-only effect.",
        ],
        "scorecard": [("1.5 s", "warning before the step"), ("0.633 AP", "pose + scene, real video"), ("133k", "real pedestrian-frames")],
        "tools": ["PyTorch", "MediaPipe", "CARLA 0.9.16", "ONNX", "OpenCV", "pytest"],
    },
    {
        "slug": "intersection-negotiation-marl",
        "title": "Two cars, one junction, no traffic lights",
        "subtitle": "Multi-agent reinforcement learning with intent inference at unsignalised intersections",
        "short": "Two self-driving cars learn to take turns at a junction with no lights or radio, by reading each other's motion.",
        "score": "12 of 12 seeds solved, 0 collisions",
        "tags": ["Multi-agent RL", "MAPPO", "Bayesian filtering"],
        "repo": "cooperative-negotiation-marl",
        "status": "Code complete with 63 passing tests. Latest experiment (v4) done.",
        "problem": [
            "Two cars reach a junction with no lights, no signs and no way to talk. Only one can go first. Both want to avoid a crash, but each would rather be the one that goes first.",
            "That makes it a general-sum game: the cars agree on safety and disagree on the order. They have to negotiate through how they move.",
        ],
        "built": [
            "The junction as a general-sum Markov game where each car has three intent actions: creep, yield and go.",
            "A particle filter in each car that estimates whether the other car is being assertive, from its motion alone.",
            "That belief added to each car's observation, so standard policy-gradient methods still apply.",
            "MAPPO (multi-agent PPO) written from scratch, with a centralised critic.",
            "Parallel environments with a check that multi-process runs reproduce single-process runs exactly.",
        ],
        "results": [
            "The experiment is a controlled A/B: same learner, same environment, same seeds, and one line of difference in the reward.",
            "Training logs showed a \"frozen follower\" that never took its turn. Removing one reward term once the other car has cleared fixed it: <b>1.000 resolution rate and 0 collisions on all 12 seeds</b>.",
        ],
        "scorecard": [("12 / 12", "seeds fully resolved"), ("0", "collisions"), ("63", "tests passing")],
        "tools": ["PyTorch", "MAPPO", "Particle filters", "NumPy", "pytest"],
    },
    {
        "slug": "robot-arm-demo-quality",
        "title": "Which human demos teach a robot arm best?",
        "subtitle": "Demo-quality ablations for a vision-language-action model (SmolVLA) on LIBERO-10",
        "short": "Scores every teleoperated demo for how well it was driven, then trains a robot-arm policy on the best, worst and random halves.",
        "score": "41.6% vs 40.0% vs 39.8% task success",
        "tags": ["Robot learning", "VLA", "Imitation learning"],
        "repo": "vla-demo-quality",
        "status": "First result in at one seed. More seeds next.",
        "problem": [
            "Robot foundation models (vision-language-action models, or VLAs) learn from people driving a robot arm by remote control. Some of those demos are hesitant, shaky or include a fumbled grasp.",
            "Does throwing out the bad demos make the robot better? This project measures it directly.",
        ],
        "built": [
            "Five driving-quality signals per demo: length, idle time, jerk, gripper toggles and path wandering.",
            "Each signal compared against other demos of the same task, then combined into one badness score.",
            "Best, worst and random halves of LIBERO-10's 379 demos, with the same task mix and demo count in each.",
            "SmolVLA (about 450M parameters) fine-tuned on one GPU with LeRobot, then tested in simulation on a Franka arm.",
        ],
        "results": [
            "Over 500 simulated episodes: best half <b>41.6%</b>, worst half <b>40.0%</b>, random half <b>39.8%</b>, all demos 40.4%.",
            "At one seed, demo quality as scored here made no measurable difference. That is a useful answer too: it says the obvious cleanup is not where the gains are.",
        ],
        "scorecard": [("500", "simulated test episodes"), ("41.6%", "best-half success"), ("~450M", "parameter VLA, one GPU")],
        "tools": ["PyTorch", "LeRobot", "SmolVLA", "LIBERO", "pixi"],
    },
    {
        "slug": "blind-quadruped-locomotion",
        "title": "A robot dog that walks blind",
        "subtitle": "Privileged-learning transformer policy for rough-terrain quadruped locomotion",
        "short": "A transformer policy that walks a four-legged robot over slopes and stairs using only what its joints feel. No camera, no lidar.",
        "score": "Teacher-student pipeline, 61 tests passing",
        "tags": ["Legged robots", "PPO", "Transformers"],
        "repo": "rough-terrain-locomotion",
        "status": "In progress. The models, training pipeline, terrain generation and evaluation are written and tested. Large-scale training in Isaac Gym is the next step.",
        "problem": [
            "A blind robot on stairs has a memory problem. From one moment you can't tell a step down from soft ground: both feel like \"the foot went lower than expected\".",
            "The robot needs to look back over its last few steps and weigh them by where it is in its gait. That makes it a sequence-modelling problem, so the policy is a transformer.",
        ],
        "built": [
            "A teacher policy, trained with PPO, that gets information the real robot never will: a height map, contact forces, true friction and mass.",
            "A causal transformer student that sees only the last 20 control steps of joint sensing and predicts the teacher's action.",
            "Two-pass training: first the student copies the teacher, then the student drives and the teacher corrects it (DAgger-style), so it learns from its own mistakes.",
            "PPO with generalised advantage estimation, written from scratch, plus latent-estimator and stacked-history baselines.",
            "Leg odometry with slip detection, and a terrain curriculum of slopes, stairs and rough ground.",
        ],
        "results": [
            "The full pipeline runs end to end and 61 tests pass.",
            "Training results will be added here once the Isaac Gym runs are done.",
        ],
        "scorecard": [("12", "motors at 50 Hz"), ("20", "steps of joint history"), ("0", "cameras used")],
        "tools": ["PyTorch", "PPO", "Isaac Gym", "NumPy", "pytest"],
    },
    {
        "slug": "robotic-pipe-sorting",
        "title": "Picking pipes out of a messy pile",
        "subtitle": "Perception and pull-direction selection for robotic sorting of pipe piles",
        "short": "An industry project: deciding which pipe a robot should pull from a jumbled pile, and in which direction.",
        "score": "Industry project, ongoing",
        "tags": ["Manipulation", "Isaac Sim", "RGB-D"],
        "repo": None,
        "status": "Ongoing, with an industry partner. Code is private.",
        "problem": [
            "Pipes in a pile lie on top of each other at every angle. Pull the wrong one, or pull in the wrong direction, and the pile jams or collapses.",
            "The robot has to choose a pipe and a pull direction from a camera view where most pipes are partly hidden.",
        ],
        "built": [
            "An Isaac Sim pipeline that builds random piles of straight, elbow and flanged pipes and renders colour, depth and masks.",
            "A physics-based evaluator that tries candidate pulls in simulation and labels them automatically.",
            "A setup to compare input representations and learning methods fairly on the same labelled data.",
        ],
        "results": ["Work in progress. Results will be added when they can be shared."],
        "scorecard": [("3", "pipe types: straight, elbow, flanged"), ("RGB-D", "plus instance masks"), ("Auto", "physics-based labels")],
        "tools": ["Isaac Sim", "PyTorch", "Python"],
    },
    {
        "slug": "realtime-detection-tensorrt",
        "title": "Making a robot's eyes 3.5x faster",
        "subtitle": "FP16 and INT8 TensorRT optimisation of YOLO11 for real-time road-user detection",
        "short": "Runs YOLO11 detection on TensorRT at three precisions and measures accuracy, speed, memory and energy for each.",
        "score": "574 FPS, 3.57x faster, -0.09 mAP",
        "tags": ["Perception", "TensorRT", "Edge deployment"],
        "repo": "yolo11-tensorrt-quantization",
        "status": "Complete.",
        "problem": [
            "A robot's perception has to run in real time on limited hardware. Lower precision (FP16, INT8) makes models faster but can cost accuracy.",
            "This project measures exactly how much speed you gain and how much accuracy you lose, on all 5,000 COCO validation images.",
        ],
        "built": [
            "YOLO11 n, s and m benchmarked on PyTorch, ONNX Runtime and TensorRT at FP32, FP16 and INT8.",
            "INT8 post-training quantisation with NVIDIA ModelOpt, including a mixed version with the first and last layers kept in FP16.",
            "Speed, GPU memory and energy per frame measured at batch size 1 on an RTX 5060 Ti.",
            "Kernel-level profiling across 21 backend and precision configurations.",
        ],
        "results": [
            "TensorRT FP16 ran YOLO11m in <b>1.74 ms (574 FPS)</b> against 6.22 ms in PyTorch FP32, a 3.57x speed-up, for only 0.09 mAP lost.",
            "Keeping the first and last layers in FP16 cut the INT8 accuracy drop from 1.93 to <b>0.63 mAP</b>, with the smallest engine (24.1 MB) and the lowest energy per frame.",
            "Profiling showed quantise/dequantise casts took 45% of INT8 kernel time, which explains why INT8 was slower than FP16 here.",
        ],
        "scorecard": [("574 FPS", "YOLO11m, TensorRT FP16"), ("3.57x", "faster than PyTorch FP32"), ("24.1 MB", "smallest engine")],
        "tools": ["TensorRT", "NVIDIA ModelOpt", "ONNX Runtime", "PyTorch", "CUDA"],
    },
    {
        "slug": "traffic-world-model",
        "title": "A car that imagines before it acts",
        "subtitle": "World-model reinforcement learning for traffic-scene anticipation",
        "short": "A small DreamerV3-style world model, written from scratch, that imagines a traffic scene 15 steps ahead to test manoeuvres before doing them.",
        "score": "4x more accurate than the baseline",
        "tags": ["World models", "Model-based RL", "MetaDrive"],
        "repo": "traffic-world-model",
        "status": "Feasibility study on one consumer GPU. Complete.",
        "problem": [
            "Before a car changes lanes it would help to know how the scene will look a second or two later. A world model learns to predict that.",
            "The question here: can this kind of model be shrunk to fit on one consumer GPU and still be useful?",
        ],
        "built": [
            "A recurrent state-space model with categorical latents, a GRU and a CNN encoder and decoder, in the style of DreamerV3. 11.2M parameters, all written from scratch.",
            "400 scripted MetaDrive driving episodes (66,144 transitions) of top-down traffic views.",
            "An imagine-then-act planner that rolls candidate manoeuvres 15 steps forward inside the model and scores them.",
        ],
        "results": [
            "On held-out episodes, imagined 15-step rollouts are <b>4.0x more accurate</b> than the persistence baseline (repeating the last frame).",
            "Planning in imagination raised episode return but not safety, which points at the reward, not the model, as the limit.",
        ],
        "scorecard": [("4.0x", "better than persistence"), ("15", "imagined steps ahead"), ("11.2M", "parameters")],
        "tools": ["PyTorch", "MetaDrive", "Mixed precision", "pixi"],
    },
    {
        "slug": "speech-gesture-grounding",
        "title": "\"Pull over there\" plus a pointing finger",
        "subtitle": "In-cabin speech and gesture grounding for driver-vehicle interaction",
        "short": "Turns what a driver says and points at into an action, and asks a question instead of guessing when the two don't agree.",
        "score": "Silent wrong actions: 37.5% to 0%",
        "tags": ["Multimodal", "Speech", "Human-robot interaction"],
        "repo": "incabin-multimodal-grounding",
        "status": "Complete. Small test set (16 cases), reported as such.",
        "problem": [
            "A driver says \"pull over there\" and points. \"There\" means nothing alone, and neither does the gesture. Only together do they pick out a spot.",
            "A wrong clarifying question is cheap. A wrong manoeuvre is not. So the system must know when it isn't sure.",
        ],
        "built": [
            "Speech transcription with Whisper and open-vocabulary matching against the camera view with CLIP.",
            "Speech and pointing fused as a product of experts, so any disagreement spreads out the belief and triggers a question.",
            "Three possible outcomes: act, ask a specific clarifying question, or reject.",
        ],
        "results": [
            "Allowed to ask, the system was right in every case and never silently did the wrong thing.",
            "Forced to act without asking, <b>37.5%</b> of its actions were silently wrong. Asking brought that to <b>0%</b>.",
        ],
        "scorecard": [("0%", "silent errors when allowed to ask"), ("37.5%", "silent errors when forced"), ("3", "outcomes: act, ask, reject")],
        "tools": ["Whisper", "CLIP", "PyTorch", "Python"],
    },
    {
        "slug": "titan-reproduction",
        "title": "Rebuilding a CVPR paper from scratch",
        "subtitle": "Reproduction and ablation of TITAN action priors for pedestrian trajectory forecasting",
        "short": "Reimplemented TITAN (CVPR 2020) from the paper alone, with every prior on its own switch so the ablation can be run row by row.",
        "score": "Within 10% of published baselines",
        "tags": ["Trajectory forecasting", "Reproduction", "PyTorch"],
        "repo": "titan-action-prior-repro",
        "status": "Code complete. Full TITAN-dataset run waiting on dataset access.",
        "problem": [
            "Papers are easy to cite and hard to reproduce. TITAN predicts where pedestrians will walk using what they are doing (an action prior), and no reference code is public.",
        ],
        "built": [
            "The action-recognition branch, GRU trajectory decoder and Agent Importance Mechanism, written from the paper.",
            "Each prior (ego motion, interaction, action) on an independent switch: seven ablation configurations.",
            "54 tests covering the full pipeline from data loader to final displacement error.",
        ],
        "results": [
            "Decoder and ADE/FDE metrics validated on the public ETH/UCY benchmark, within 10% of published baselines.",
            "In a seven-configuration ablation, the action prior was the strongest on its own (8.24 px ADE better than vanilla), and combining priors made it worse.",
        ],
        "scorecard": [("7", "ablation configurations"), ("<10%", "from published ETH/UCY baselines"), ("54", "tests passing")],
        "tools": ["PyTorch", "3D CNNs", "GRU", "pytest"],
    },
]

# ---------------------------------------------------------------------------------------
# SIDE QUESTS: other things you've built (one line each).
# ---------------------------------------------------------------------------------------
SIDE_QUESTS = [
    ("tinyserve", "LLM inference server", "Continuous batching and a paged KV cache for Qwen2.5-1.5B: 2,948 tokens/s, 7.6x faster than Hugging Face static batching."),
    ("hnswlite", "Vector search in C++", "HNSW nearest-neighbour search from scratch with Python bindings: 14% faster than hnswlib at the same recall on SIFT-1M."),
    ("streamlog", "Kafka-style log in Go", "4.57M messages/s with fsync before every ack. Killed 200 times with kill -9 and lost 0 acknowledged records."),
    ("Scout", "Product search engine in Go", "BM25F search over 1.2M products. Beats SQLite FTS5 on 22k human-labelled queries at 0.42 ms median latency."),
    ("Tollgate", "API gateway in Go", "Load balancing, health checks and cluster-wide rate limits, adding only 475 ns per request at 110k requests/s."),
    ("Weave", "Collaborative text editor", "A sequence CRDT written from scratch with live cursors and offline edits. Replays edits 3.5x faster than Yjs."),
    ("fleet-telemetry-pipeline", "Vehicle telemetry pipeline", "ETL of 8.3M vehicle records into PostgreSQL. SQL alerts caught 22/22 tyre leaks with no false alarms."),
    ("incabin-rag-lora", "Car-manual assistant", "A small LLM that answers from a car manual and says so when it can't. Training it to abstain cut silent errors from 80% to about 5%."),
]

# ---------------------------------------------------------------------------------------
# CAREER: experience and education.
# ---------------------------------------------------------------------------------------
CAREER = [
    {
        "when": "Jul 2026 – now",
        "role": "Project Scientist",
        "org": "Department of Ocean Engineering, IIT Madras",
        "where": "Chennai, India",
        "points": [
            "Lead the simulation workstream of the lab's autonomy programme: perception, state estimation and real-vehicle trials.",
            "Lead author of the lab's IEEE Robotics and Automation Letters (RA-L) paper on its physics-based robot simulator.",
            "Built two small autonomous vehicle prototypes end to end, from wiring and thruster bring-up to field runs, and took the control stack from simulation through hardware-in-the-loop to real trials.",
            "Teaching assistant for Ship Drawing and Calculations: run the weekly labs and grade all submissions.",
        ],
    },
    {
        "when": "Mar 2026 – Apr 2026",
        "role": "Robotics Intern",
        "org": "Pravahan Engineering Pvt. Ltd.",
        "where": "India",
        "points": [
            "Implemented an Extended Kalman Filter fusing GNSS, IMU and Lidar, and tuned it against measured ground truth.",
            "Compared PID and MPC heading control across more than 20 disturbance runs in simulation.",
            "Designed the autonomy stack so ROS 2 handled perception and sensing while a behaviour layer arbitrated decisions.",
        ],
    },
    {
        "when": "Dec 2024 – now",
        "role": "Research Assistant",
        "org": "Prof. Suresh Rajendran, IIT Madras",
        "where": "Chennai, India",
        "points": [
            "Built a perception pipeline for a degraded sensor (imaging sonar): denoising, object detection with an adapted YOLO, and 3D reconstruction.",
            "Benchmarked self-supervised denoisers (Noise2Noise, Neighbor2Neighbor) against supervised and do-nothing baselines.",
            "Found that a 7 dB gain in image quality came with a 57x drop in detection overlap: prettier images were worse for the detector.",
        ],
    },
    {
        "when": "2022 – 2026",
        "role": "B.Tech, Naval Architecture and Ocean Engineering",
        "org": "Indian Institute of Technology Madras",
        "where": "Chennai, India",
        "points": [
            "Coursework in autonomous vehicles, control, probability and stochastic processes, and instrumentation.",
        ],
    },
]

# ---------------------------------------------------------------------------------------
# SKILLS ("inventory")
# ---------------------------------------------------------------------------------------
SKILLS = [
    ("Languages", ["Python", "C++", "C#", "Go", "TypeScript", "C", "SQL", "MATLAB"]),
    ("Robotics", ["ROS 2", "Gazebo", "Isaac Sim", "MuJoCo", "CARLA", "MetaDrive", "Unity", "LeRobot"]),
    ("Robot learning", ["PyTorch", "PPO / MAPPO", "Imitation learning", "World models", "Transformers", "Gymnasium"]),
    ("Perception", ["OpenCV", "Open3D", "YOLO", "MediaPipe", "CLIP", "Whisper"]),
    ("Estimation & control", ["Extended Kalman Filter", "Particle filters", "PID", "MPC", "Leg odometry"]),
    ("Deployment", ["TensorRT", "ONNX", "CUDA", "Docker", "Linux", "Git", "pytest"]),
]

# ---------------------------------------------------------------------------------------
# TROPHIES (achievements)
# ---------------------------------------------------------------------------------------
TROPHIES = [
    ("Patent filed", "Indian provisional patent 202641100952 (Aug 2026): a physics-based navigation system and method, from my simulator work."),
    ("Journal paper in progress", "Leading the lab's IEEE RA-L paper on the simulator's dynamics validation and sensor models."),
    ("JEE Advanced 2022", "Qualified among 1.5 lakh+ candidates through self-study, with no coaching."),
    ("GATE 2026", "Qualified, alongside final-year coursework."),
]

# =======================================================================================
# Templates. You shouldn't need to edit below this line.
# =======================================================================================

BASE = PROFILE["base_url"]
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&'
         'family=JetBrains+Mono:wght@400;500;600&family=Pixelify+Sans:wght@500;600&display=swap">')

BOT_SVG = """<svg class="bot" viewBox="0 0 16 16" shape-rendering="crispEdges" role="img" aria-label="Pixel-art robot">
<g class="body">
<rect x="7.5" y="0" width="1" height="2" fill="#b0add6"/><rect x="7" y="0" width="2" height="1" fill="#ff7ab6"/>
<rect x="3" y="2" width="10" height="7" fill="#d9d6ff"/><rect x="2" y="4" width="1" height="3" fill="#b0add6"/><rect x="13" y="4" width="1" height="3" fill="#b0add6"/>
<rect x="4" y="3" width="8" height="5" fill="#1c1840"/>
<g class="eyes"><rect x="5" y="4" width="2" height="2" fill="#7fe6f2"/><rect x="9" y="4" width="2" height="2" fill="#7fe6f2"/></g>
<rect x="6" y="7" width="4" height="0.6" fill="#ff7ab6"/>
<rect x="4" y="10" width="8" height="4" fill="#7fe6f2"/><rect x="6" y="11" width="4" height="2" fill="#1c1840"/><rect x="7" y="11.5" width="1" height="1" fill="#ffd36e"/><rect x="8.5" y="11.5" width="1" height="1" fill="#8ef0a8"/>
<rect x="2" y="10.5" width="2" height="1.2" fill="#d9d6ff"/><rect x="12" y="10.5" width="2" height="1.2" fill="#d9d6ff"/>
<rect x="5" y="14" width="2" height="2" fill="#d9d6ff"/><rect x="9" y="14" width="2" height="2" fill="#d9d6ff"/>
</g></svg>"""

TROPHY_SVG = """<svg viewBox="0 0 12 12" shape-rendering="crispEdges" aria-hidden="true">
<rect x="3" y="1" width="6" height="5" fill="#ffd36e"/><rect x="1" y="2" width="2" height="2" fill="#ffd36e"/><rect x="9" y="2" width="2" height="2" fill="#ffd36e"/>
<rect x="4" y="2" width="1" height="2" fill="#fff3c4"/><rect x="5" y="6" width="2" height="2" fill="#e0a83a"/><rect x="3" y="8" width="6" height="2" fill="#e0a83a"/></svg>"""


def esc(s):
    return html.escape(s, quote=True)


def page(title, description, path, body, depth=0, jsonld=None):
    """Wrap a page body in the full HTML document with SEO and social tags."""
    up = "../" * depth
    url = BASE + path
    ld = ""
    if jsonld:
        ld = '<script type="application/ld+json">' + json.dumps(jsonld, ensure_ascii=False, indent=1) + "</script>"
    email_link = f'<a href="mailto:{esc(PROFILE["email"])}">Email</a>' if PROFILE["email"] else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<meta name="author" content="{esc(PROFILE['name'])}">
<link rel="canonical" href="{url}">
<meta property="og:type" content="website">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{BASE}assets/og.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#14112a">
<link rel="icon" href="{up}assets/favicon.svg" type="image/svg+xml">
<link rel="alternate" type="text/plain" href="{BASE}llms.txt" title="Plain-text profile for AI assistants">
{FONTS}
<link rel="stylesheet" href="{up}assets/style.css">
{ld}
</head>
<body>
<header class="hud">
  <div class="wrap">
    <a class="logo" href="{up}index.html"><span class="p1">P1</span>{esc(PROFILE['short_name'].upper())}</a>
    <nav aria-label="Main">
      <a href="{up}index.html#levels">Projects</a>
      <a href="{up}index.html#side-quests">Side quests</a>
      <a href="{up}index.html#career">Experience</a>
      <a href="{up}index.html#inventory">Skills</a>
      <a href="{up}index.html#contact">Contact</a>
      <a class="resume" href="{up}resume.html">Resume</a>
    </nav>
  </div>
</header>
<main>
{body}
</main>
<footer>
  <div class="wrap">
    <span>&copy; 2026 {esc(PROFILE['name'])} &middot; {esc(PROFILE['location'])}</span>
    <span><a href="{PROFILE['github']}">GitHub</a> &middot; <a href="{PROFILE['linkedin']}">LinkedIn</a>{' &middot; ' + email_link if email_link else ''}</span>
  </div>
</footer>
</body>
</html>
"""


def person_ld():
    same = [PROFILE["github"], PROFILE["linkedin"]]
    ld = {
        "@context": "https://schema.org",
        "@type": "ProfilePage",
        "url": BASE,
        "mainEntity": {
            "@type": "Person",
            "name": PROFILE["name"],
            "alternateName": PROFILE["other_names"],
            "jobTitle": PROFILE["title"],
            "description": PROFILE["headline"] + ". " + PROFILE["intro"],
            "url": BASE,
            "image": BASE + "assets/og.png",
            "address": {"@type": "PostalAddress", "addressLocality": "Chennai", "addressCountry": "IN"},
            "alumniOf": {"@type": "CollegeOrUniversity", "name": PROFILE["school"]},
            "affiliation": {"@type": "CollegeOrUniversity", "name": PROFILE["school"]},
            "knowsAbout": sorted({s for _, items in SKILLS for s in items} | {
                "Robotics", "Robot perception", "Robot learning", "Reinforcement learning",
                "Multi-agent reinforcement learning", "Legged locomotion", "Sensor fusion",
                "State estimation", "Autonomous driving", "Computer vision"}),
            "sameAs": same,
        },
    }
    if PROFILE["email"]:
        ld["mainEntity"]["email"] = "mailto:" + PROFILE["email"]
    return ld


def build_index():
    stats = "".join(f'<span class="chip{" gold" if i == 0 else ""}">{esc(s)}</span>' for i, s in enumerate([
        "Patent filed, 2026", "B.Tech, IIT Madras", "9 robotics projects", "Open to robotics roles"]))
    email_btn = f'<a class="btn" href="mailto:{esc(PROFILE["email"])}">Email</a>' if PROFILE["email"] else ""

    levels = []
    for i, p in enumerate(PROJECTS, 1):
        tags = "".join(f"<span>{esc(t)}</span>" for t in p["tags"])
        levels.append(f"""<a class="level" href="projects/{p['slug']}.html">
  <span class="label">Level {i:02d}</span>
  <h3>{esc(p['title'])}</h3>
  <p>{esc(p['short'])}</p>
  <div class="tags">{tags}</div>
  <span class="score">&#9733; {esc(p['score'])}</span>
</a>""")

    quests = "".join(f"""<div class="quest"><h3><a href="{PROFILE['github']}/{repo}">{esc(name)}</a><small>{esc(repo)}</small></h3><p>{esc(desc)}</p></div>"""
                     for repo, name, desc in SIDE_QUESTS)

    jobs = []
    for j in CAREER:
        pts = "".join(f"<li>{esc(x)}</li>" for x in j["points"])
        jobs.append(f"""<article class="job">
  <div><div class="when">{esc(j['when'])}</div><div class="where">{esc(j['where'])}</div></div>
  <div><h3>{esc(j['role'])}</h3><div class="org">{esc(j['org'])}</div><ul>{pts}</ul></div>
</article>""")

    slots = "".join(f"""<div class="slot"><h3>{esc(group)}</h3><ul>{''.join(f'<li>{esc(s)}</li>' for s in items)}</ul></div>"""
                    for group, items in SKILLS)

    trophies = "".join(f"""<div class="trophy">{TROPHY_SVG}<div><h3>{esc(t)}</h3><p>{esc(d)}</p></div></div>""" for t, d in TROPHIES)

    body = f"""
<section class="hero" id="about">
  <div class="wrap">
    {BOT_SVG}
    <div>
      <span class="label">Player 1 &middot; Ready</span>
      <h1>{esc(PROFILE['name'])}</h1>
      <p class="class">Class: {esc(PROFILE['title'])} &nbsp;&middot;&nbsp; Base: IIT Madras, Chennai</p>
      <p class="intro">{esc(PROFILE['intro'])}</p>
      <div class="stats">{stats}</div>
      <div class="buttons">
        <a class="btn primary" href="#levels">Select a level</a>
        <a class="btn" href="resume.html">Resume</a>
        <a class="btn" href="{PROFILE['github']}">GitHub</a>
        <a class="btn" href="{PROFILE['linkedin']}">LinkedIn</a>
        {email_btn}
      </div>
    </div>
  </div>
</section>

<section class="block" id="levels">
  <div class="wrap">
    <div class="block-head">
      <span class="label">Select a level</span>
      <h2>Robotics projects</h2>
      <p>Robots that see, walk, grab things and share the road. Each level has its own page with the problem, what I built and the results.</p>
    </div>
    <div class="levels">
      {''.join(levels)}
    </div>
  </div>
</section>

<section class="block" id="side-quests">
  <div class="wrap">
    <div class="block-head">
      <span class="label">Side quests</span>
      <h2>Other things I've built</h2>
      <p>Systems and ML projects, each written from scratch and benchmarked against well-known tools.</p>
    </div>
    <div class="quests">{quests}</div>
  </div>
</section>

<section class="block" id="career">
  <div class="wrap">
    <div class="block-head">
      <span class="label">Career mode</span>
      <h2>Experience and education</h2>
    </div>
    <div class="career">{''.join(jobs)}</div>
  </div>
</section>

<section class="block" id="inventory">
  <div class="wrap">
    <div class="block-head">
      <span class="label">Inventory</span>
      <h2>Skills and tools</h2>
    </div>
    <div class="inventory">{slots}</div>
  </div>
</section>

<section class="block" id="trophies">
  <div class="wrap">
    <div class="block-head">
      <span class="label">Trophies</span>
      <h2>Achievements</h2>
    </div>
    <div class="trophies">{trophies}</div>
  </div>
</section>

<section class="block contact" id="contact">
  <div class="wrap">
    <span class="label">Player 2 wanted</span>
    <h2 class="big">Continue? <span>Let's build robots.</span></h2>
    <p>{esc(PROFILE['open_to'])}. The quickest way to reach me is LinkedIn.</p>
    <div class="links">
      <a class="btn primary" href="{PROFILE['linkedin']}">Message me on LinkedIn</a>
      <a class="btn" href="{PROFILE['github']}">GitHub</a>
      <a class="btn" href="resume.pdf">Download resume (PDF)</a>
      {email_btn}
    </div>
  </div>
</section>
"""
    title = f"{PROFILE['name']} | Robotics Engineer"
    desc = (f"{PROFILE['name']} is a robotics engineer at IIT Madras working on robot perception, "
            "robot learning, legged locomotion, multi-agent RL and sensor fusion.")
    return page(title, desc, "", body, 0, person_ld())


def build_project(i, p):
    prev_p = PROJECTS[i - 2] if i > 1 else None
    next_p = PROJECTS[i] if i < len(PROJECTS) else None
    repo_url = f"{PROFILE['github']}/{p['repo']}" if p["repo"] else None
    meta = "".join(f'<span class="chip">{esc(t)}</span>' for t in p["tags"])
    if repo_url:
        meta += f'<a class="chip gold" href="{repo_url}">Code on GitHub &rarr;</a>'
    problem = "".join(f"<p>{esc(x)}</p>" for x in p["problem"])
    built = "".join(f"<li>{esc(x)}</li>" for x in p["built"])
    results = "".join(f"<li>{x}</li>" for x in p["results"])  # results may contain <b>
    tools = "".join(f'<span class="chip">{esc(t)}</span>' for t in p["tools"])
    score = "".join(f"<li><b>{esc(v)}</b>{esc(l)}</li>" for v, l in p["scorecard"])
    nav = '<div class="next-prev">'
    nav += f'<a href="{prev_p["slug"]}.html">&larr; Level {i-1:02d}: {esc(prev_p["title"])}</a>' if prev_p else '<span></span>'
    nav += f'<a href="{next_p["slug"]}.html">Level {i+1:02d}: {esc(next_p["title"])} &rarr;</a>' if next_p else '<a href="../index.html#levels">Back to all levels &rarr;</a>'
    nav += "</div>"

    body = f"""
<div class="wrap">
  <div class="crumbs"><a href="../index.html#levels">&larr; Level select</a></div>
  <div class="proj-head">
    <span class="label">Level {i:02d}</span>
    <h1>{esc(p['title'])}</h1>
    <p class="lede">{esc(p['subtitle'])}. {esc(p['short'])}</p>
    <div class="meta">{meta}</div>
  </div>
  <div class="proj-body">
    <div class="prose">
      <div><h2>Status</h2><p class="status">{esc(p['status'])}</p></div>
      <div><h2>The problem</h2>{problem}</div>
      <div><h2>What I built</h2><ul>{built}</ul></div>
      <div><h2>Results</h2><ul>{results}</ul></div>
      <div><h2>Tools</h2><div class="tools">{tools}</div></div>
    </div>
    <aside class="scorecard" aria-label="Key numbers">
      <span class="label">High scores</span>
      <ul>{score}</ul>
    </aside>
  </div>
  {nav}
</div>
"""
    ld = {
        "@context": "https://schema.org",
        "@type": "SoftwareSourceCode" if repo_url else "CreativeWork",
        "name": p["subtitle"],
        "alternateName": p["title"],
        "description": p["short"],
        "url": BASE + f"projects/{p['slug']}.html",
        "author": {"@type": "Person", "name": PROFILE["name"], "url": BASE},
        "keywords": ", ".join(p["tags"] + p["tools"]),
    }
    if repo_url:
        ld["codeRepository"] = repo_url
    title = f"{p['title']} | {PROFILE['name']}"
    return page(title, f"{p['subtitle']}. {p['short']}", f"projects/{p['slug']}.html", body, 1, ld)


def build_resume():
    jobs = "".join(f"""<h3>{esc(j['role'])}, {esc(j['org'])}</h3><div class="sub">{esc(j['when'])} &middot; {esc(j['where'])}</div>
<ul>{''.join(f'<li>{esc(x)}</li>' for x in j['points'])}</ul>""" for j in CAREER)
    projects = "".join(f"""<h3><a href="projects/{p['slug']}.html">{esc(p['subtitle'])}</a></h3>
<p>{esc(p['short'])} Result: {esc(p['score'])}.</p>""" for p in PROJECTS)
    quests = "".join(f"<li><b>{esc(n)}</b>: {esc(d)}</li>" for _, n, d in SIDE_QUESTS)
    skills = "".join(f"<li><b>{esc(g)}:</b> {esc(', '.join(items))}</li>" for g, items in SKILLS)
    trophies = "".join(f"<li><b>{esc(t)}:</b> {esc(d)}</li>" for t, d in TROPHIES)
    contact = f'<a href="{PROFILE["linkedin"]}">LinkedIn</a> &middot; <a href="{PROFILE["github"]}">GitHub</a>'
    if PROFILE["email"]:
        contact += f' &middot; {esc(PROFILE["email"])}'
    body = f"""
<div class="wrap"><article class="resume-page">
  <span class="label">Resume &middot; plain text</span>
  <h1>{esc(PROFILE['name'])}</h1>
  <p>{esc(PROFILE['headline'])}. {esc(PROFILE['location'])}.</p>
  <p class="sub">{contact} &middot; <a href="resume.pdf">PDF version</a></p>
  <h2>Summary</h2><p>{esc(PROFILE['intro'])}</p>
  <h2>Experience and education</h2>{jobs}
  <h2>Robotics projects</h2>{projects}
  <h2>Other projects</h2><ul>{quests}</ul>
  <h2>Skills</h2><ul>{skills}</ul>
  <h2>Achievements</h2><ul>{trophies}</ul>
</article></div>
"""
    return page(f"Resume | {PROFILE['name']}", f"Resume of {PROFILE['name']}, {PROFILE['headline'].lower()}.", "resume.html", body, 0, person_ld())


def build_404():
    body = f"""<div class="wrap" style="padding-block:96px">
<span class="label">Game over</span>
<h1 style="font-size:clamp(34px,6vw,56px);margin-top:10px">404: level not found</h1>
<p style="color:var(--muted);margin-top:14px">This page doesn't exist. Insert coin to continue.</p>
<div class="buttons"><a class="btn primary" href="{BASE}">Continue &rarr;</a></div></div>"""
    return page(f"Page not found | {PROFILE['name']}", "Page not found.", "404.html", body, 0)


def build_llms_txt():
    lines = [
        f"# {PROFILE['name']}",
        "",
        f"> {PROFILE['headline']}. Based in {PROFILE['location']}. {PROFILE['open_to']}.",
        "",
        PROFILE["intro"],
        "",
        "Also known as: " + ", ".join(PROFILE["other_names"]) + ".",
        "",
        "## Key facts",
        f"- Current role: Project Scientist, Department of Ocean Engineering, IIT Madras (since July 2026)",
        f"- Education: {PROFILE['degree']}",
        "- Indian provisional patent 202641100952 (Aug 2026)",
        "- Internship: Robotics Intern, Pravahan Engineering (EKF sensor fusion of GNSS, IMU and Lidar; PID vs MPC control)",
        "- Research: perception under degraded sensing (denoising, detection, 3D reconstruction)",
        "- Skills: " + "; ".join(f"{g}: {', '.join(items)}" for g, items in SKILLS),
        "",
        "## Robotics projects",
    ]
    for p in PROJECTS:
        lines.append(f"- [{p['subtitle']}]({BASE}projects/{p['slug']}.html): {p['short']} Result: {p['score']}.")
    lines += ["", "## Other projects"]
    for repo, name, desc in SIDE_QUESTS:
        lines.append(f"- [{name}]({PROFILE['github']}/{repo}): {desc}")
    lines += ["", "## Links",
              f"- Website: {BASE}",
              f"- Plain-text resume: {BASE}resume.html",
              f"- GitHub: {PROFILE['github']}",
              f"- LinkedIn: {PROFILE['linkedin']}"]
    if PROFILE["email"]:
        lines.append(f"- Email: {PROFILE['email']}")
    return "\n".join(lines) + "\n"


def build_sitemap():
    urls = [BASE, BASE + "resume.html"] + [BASE + f"projects/{p['slug']}.html" for p in PROJECTS]
    items = "".join(f"<url><loc>{u}</loc></url>" for u in urls)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{items}</urlset>\n'


ROBOTS = """# Everyone is welcome, including AI crawlers and assistants.
User-agent: *
Allow: /

Sitemap: {base}sitemap.xml
"""

FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" shape-rendering="crispEdges">
<rect width="16" height="16" rx="3" fill="#14112a"/><rect x="3" y="3" width="10" height="8" fill="#d9d6ff"/>
<rect x="4" y="4" width="8" height="6" fill="#1c1840"/><rect x="5" y="5" width="2" height="2" fill="#7fe6f2"/>
<rect x="9" y="5" width="2" height="2" fill="#7fe6f2"/><rect x="6" y="8" width="4" height="1" fill="#ff7ab6"/>
<rect x="7.5" y="1" width="1" height="2" fill="#ff7ab6"/><rect x="5" y="12" width="6" height="2" fill="#7fe6f2"/></svg>
"""


def main():
    (ROOT / "projects").mkdir(exist_ok=True)
    (ROOT / "assets").mkdir(exist_ok=True)
    (ROOT / "index.html").write_text(build_index(), encoding="utf-8")
    for i, p in enumerate(PROJECTS, 1):
        (ROOT / "projects" / f"{p['slug']}.html").write_text(build_project(i, p), encoding="utf-8")
    (ROOT / "resume.html").write_text(build_resume(), encoding="utf-8")
    (ROOT / "404.html").write_text(build_404(), encoding="utf-8")
    (ROOT / "llms.txt").write_text(build_llms_txt(), encoding="utf-8")
    (ROOT / "sitemap.xml").write_text(build_sitemap(), encoding="utf-8")
    (ROOT / "robots.txt").write_text(ROBOTS.format(base=BASE), encoding="utf-8")
    (ROOT / "assets" / "favicon.svg").write_text(FAVICON, encoding="utf-8")
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built {len(PROJECTS)} project pages plus index, resume, 404, llms.txt, sitemap and robots.txt.")


if __name__ == "__main__":
    main()
