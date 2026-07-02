import { API_BASE_URL } from "../config";
import type { ModelInfo, PredictionResponse, ScanMetadata } from "../types";

type UploadImage = {
  uri: string;
  name?: string;
  mimeType?: string;
};

export class ApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function parseJsonOrThrow<T>(response: Response, fallbackMessage: string): Promise<T> {
  let payload: unknown = null;

  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? String((payload as { detail?: unknown }).detail)
        : fallbackMessage;
    throw new ApiError(detail);
  }

  return payload as T;
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

export async function getModelInfo(): Promise<ModelInfo | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/model-info`);
    return response.ok ? ((await response.json()) as ModelInfo) : null;
  } catch {
    return null;
  }
}

export async function predictImage(image: UploadImage, metadata: ScanMetadata): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("image", {
    uri: image.uri,
    name: image.name ?? "skin-image.jpg",
    type: image.mimeType ?? "image/jpeg"
  } as unknown as Blob);

  Object.entries(metadata).forEach(([key, value]) => {
    const trimmed = value.trim();
    if (trimmed) {
      formData.append(key, trimmed);
    }
  });

  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    body: formData,
    headers: {
      Accept: "application/json"
    }
  });

  return parseJsonOrThrow<PredictionResponse>(response, "The image could not be analyzed.");
}
