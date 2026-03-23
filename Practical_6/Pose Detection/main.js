const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const poseText = document.getElementById("poseText");
const gestureText = document.getElementById("gestureText");

let detector;

// CAMERA SETUP
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
    });

    video.srcObject = stream;

    await new Promise(r => video.onloadedmetadata = r);
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

// DRAW KEYPOINTS
function drawPoints(keypoints) {
    keypoints.forEach(k => {
        if (k.score > 0.4) {
            ctx.beginPath();
            ctx.arc(k.x, k.y, 6, 0, 2 * Math.PI);
            ctx.fillStyle = "#00ffff";
            ctx.shadowBlur = 10;
            ctx.shadowColor = "#00ffff";
            ctx.fill();
        }
    });
}

// DRAW SKELETON
function drawSkeleton(keypoints) {
    const pairs = poseDetection.util.getAdjacentPairs(
        poseDetection.SupportedModels.MoveNet
    );

    pairs.forEach(([i, j]) => {
        const a = keypoints[i];
        const b = keypoints[j];

        if (a.score > 0.4 && b.score > 0.4) {
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = "#ffcc00";
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
}

// ANGLE CALCULATION
function getAngle(A, B, C) {
    const AB = { x: A.x - B.x, y: A.y - B.y };
    const CB = { x: C.x - B.x, y: C.y - B.y };

    const dot = AB.x * CB.x + AB.y * CB.y;
    const magAB = Math.sqrt(AB.x**2 + AB.y**2);
    const magCB = Math.sqrt(CB.x**2 + CB.y**2);

    return Math.acos(dot / (magAB * magCB)) * (180 / Math.PI);
}

// POSE DETECTION (FIXED)
function detectPoseType(keypoints) {
    const get = (name) => keypoints.find(k => k.name === name);

    const ls = get("left_shoulder");
    const rs = get("right_shoulder");
    const lw = get("left_wrist");
    const rw = get("right_wrist");
    const lh = get("left_hip");
    const lk = get("left_knee");
    const la = get("left_ankle");

    if (!ls || !lh || !lk || !la) return "Detecting...";

    // Hands Up
    if (lw && rw && lw.y < ls.y && rw.y < rs.y)
        return "🙌 Hands Up";

    // T Pose
    if (lw && rw &&
        Math.abs(lw.y - ls.y) < 30 &&
        Math.abs(rw.y - rs.y) < 30)
        return "🧍 T-Pose";

    const kneeAngle = getAngle(lh, lk, la);

    // Sitting
    if (kneeAngle < 90 && lh.y > ls.y + 100)
        return "🪑 Sitting";

    // Squat
    if (kneeAngle > 90 && kneeAngle < 140)
        return "🏋️ Squat";

    return "🧍 Standing";
}

// GESTURE CONTROL
function detectGesture(poseName) {
    if (poseName === "🙌 Hands Up") return "▶️ Play / Pause";
    if (poseName === "🧍 T-Pose") return "📂 Open App";
    if (poseName === "🏋️ Squat") return "🔊 Volume Down";
    if (poseName === "🪑 Sitting") return "🔇 Mute";

    return "Idle";
}

// START
async function start() {
    await setupCamera();

    detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
        }
    );

    detect();
}

// LOOP
async function detect() {

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
        const poses = await detector.estimatePoses(video);

        poses.forEach(p => {
            drawPoints(p.keypoints);
            drawSkeleton(p.keypoints);

            const poseName = detectPoseType(p.keypoints);
            const gesture = detectGesture(poseName);

            poseText.innerText = "Pose: " + poseName;
            gestureText.innerText = "Gesture: " + gesture;
        });

    } catch (err) {
        console.error(err);
    }

    requestAnimationFrame(detect);
}