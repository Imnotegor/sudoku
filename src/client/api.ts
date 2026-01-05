import type { ScanResponse } from "./types";

/** Error thrown when API request fails */
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Scan an image and get the Sudoku grid */
export async function scanImage(file: File): Promise<ScanResponse> {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch("/api/scan", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new ApiError(errorText || "Scan failed", response.status);
  }

  const data = await response.json();

  if (!data?.puzzle || !data?.solution) {
    throw new ApiError("Invalid response from server");
  }

  return data as ScanResponse;
}
