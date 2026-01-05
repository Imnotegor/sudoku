import { t } from "../i18n";
import { showToast } from "./toast";

interface CameraElements {
  camera: HTMLElement;
  video: HTMLVideoElement;
  captureBtn: HTMLButtonElement;
  closeBtn: HTMLButtonElement;
  canvas: HTMLCanvasElement;
}

let elements: CameraElements | null = null;
let stream: MediaStream | null = null;

/** Initialize the camera module with the camera elements */
export function initCamera(
  camera: HTMLElement,
  video: HTMLVideoElement,
  captureBtn: HTMLButtonElement,
  closeBtn: HTMLButtonElement,
  canvas: HTMLCanvasElement
): void {
  elements = { camera, video, captureBtn, closeBtn, canvas };
}

/** Check if camera is supported */
export function isCameraSupported(): boolean {
  return Boolean(navigator.mediaDevices?.getUserMedia);
}

/** Open the camera */
export async function openCamera(): Promise<boolean> {
  if (!elements) return false;

  if (!isCameraSupported()) {
    showToast(t("toast_camera_unsupported"));
    return false;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    elements.video.srcObject = stream;
    elements.camera.hidden = false;
    return true;
  } catch {
    showToast(t("toast_camera_denied"));
    return false;
  }
}

/** Close the camera */
export function closeCamera(): void {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  if (elements) {
    elements.video.srcObject = null;
    elements.camera.hidden = true;
  }
}

/** Capture an image from the camera */
export function captureImage(): File | null {
  if (!stream || !elements) return null;

  const video = elements.video;
  const canvas = elements.canvas;

  if (!video.videoWidth) {
    showToast(t("toast_camera_denied"));
    return null;
  }

  // Capture a square from the center of the video
  const size = Math.min(video.videoWidth, video.videoHeight);
  const sx = (video.videoWidth - size) / 2;
  const sy = (video.videoHeight - size) / 2;

  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.drawImage(video, sx, sy, size, size, 0, 0, size, size);

  // Convert canvas to blob synchronously by using toDataURL
  const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
  const byteString = atob(dataUrl.split(",")[1]);
  const mimeString = dataUrl.split(",")[0].split(":")[1].split(";")[0];
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }
  const blob = new Blob([ab], { type: mimeString });
  return new File([blob], "capture.jpg", { type: "image/jpeg" });
}

/** Bind camera event handlers */
export function bindCameraEvents(onCapture: (file: File) => void): void {
  if (!elements) return;

  elements.closeBtn.addEventListener("click", closeCamera);
  elements.captureBtn.addEventListener("click", () => {
    const file = captureImage();
    if (file) {
      onCapture(file);
    }
    closeCamera();
  });
}
