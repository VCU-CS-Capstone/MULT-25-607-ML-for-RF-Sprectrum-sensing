// Utility function to validate PSD data
export const validatePSDData = (psd) => {
  if (!Array.isArray(psd)) return false;
  if (psd.length !== 2048) return false;
  return true;
};

// Utility function to create image data from PSD values
export const createRowImageData = (
  width,
  downsampled,
  colorMap,
  globalOffset,
  detections,
  minDbm = -110,
  maxDbm = -70,
) => {
  const imgData = new ImageData(width, 1);

  for (let i = 0; i < width; i++) {
    // Normalize the dBm value to a float between 0 and 1.
    // For example, -180 maps to 0 and -40 maps to 1.
    const normalized = (downsampled[i] - minDbm) / (maxDbm - minDbm);
    // Clamp normalized value between 0 and 1 (in case of out-of-bound values).
    const clampedNormalized = Math.max(0, Math.min(1, normalized));
    // Convert normalized value to an intensity in the range [0, 255].
    const baseIntensity = Math.floor(clampedNormalized * 255);

    let offset = globalOffset;


    // Add offset and clamp intensity to [0, 255].
    const clampedIntensity = Math.max(0, Math.min(255, baseIntensity + offset));
    const colorEntry = colorMap[clampedIntensity] || [0, 0, 0, 255];
    const [r, g, b, a] = colorEntry;

    // Set pixel values in the image data.
    const idx = i * 4;
    imgData.data[idx] = r;
    imgData.data[idx + 1] = g;
    imgData.data[idx + 2] = b;
    if (detections[0] < i & detections[1] > i) {
      imgData.data[idx + 3] = a;
    } else {
      imgData.data[idx + 3] = a / 2;
    }
  }

  return imgData;
};

// Utility function to shift image data up
export const shiftImageUp = (ctx, width, height) => {
  try {
    const image = ctx.getImageData(0, 1, width, height - 1);
    ctx.putImageData(image, 0, 0);
    return true;
  } catch (error) {
    console.error("Error shifting image data:", error);
    return false;
  }
};

// Main drawing function
export const drawSpectrogram = ({
  canvas,
  psd,
  colorMap,
  globalOffset,
  detections,
  onError = () => { },
  drawTimeout = null,
  setDrawTimeout = () => { },
}) => {
  return new Promise((resolve, reject) => {
    try {
      // Validate inputs
      if (!canvas) {
        reject(new Error("Canvas not provided"));
        return;
      }

      if (!validatePSDData(psd)) {
        onError("Invalid PSD data format.");
        reject(new Error("Invalid PSD data format"));
        return;
      }

      // Setup canvas context
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      const width = canvas.width;
      const height = canvas.height;
      const psdToCanvasFactor = psd.length / width;

      // Check if previous draw is still in progress
      if (drawTimeout) {
        resolve();
        return;
      }

      // Schedule drawing
      setDrawTimeout(
        requestAnimationFrame(() => {
          // Downsample PSD data to match canvas width
          const downsampled = downsamplePsd(psd, width);

          // Create image data for new row
          const imgData = createRowImageData(
            width,
            downsampled,
            colorMap,
            globalOffset,
            detections,
          );

          // Shift existing image up
          const shiftSuccess = shiftImageUp(ctx, width, height);
          if (!shiftSuccess) {
            onError("Error shifting image data");
          }

          // Draw new row at the bottom
          ctx.putImageData(imgData, 0, height - 1);

          // Clear timeout and resolve
          setDrawTimeout(null);
          resolve();
        }),
      );
    } catch (error) {
      console.error("Error in drawSpectrogram:", error);
      onError("Error drawing spectrogram.");
      reject(error);
    }
  });
};

// Helper function to downsample PSD data
export const downsamplePsd = (psd, targetWidth) => {
  const step = Math.floor(psd.length / targetWidth);
  const downsampled = [];

  for (let i = 0; i < targetWidth; i++) {
    const segment = psd.slice(i * step, (i + 1) * step);
    const avg = segment.reduce((a, b) => a + b, 0) / step;
    downsampled.push(avg);
  }

  return downsampled;
};

export const clearCanvas = (ctx, width, height) => {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, width, height);
};

export const createSpectrogramDrawer = (canvas, colorMap, onError, lowerDbm, upperDbm) => {
  let currentDrawTimeout = null;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const draw = (psd, globalOffset, detections,) => {
    return drawSpectrogram({
      canvas,
      psd,
      colorMap,
      globalOffset,
      detections,
      onError,
      drawTimeout: currentDrawTimeout,
      setDrawTimeout: (timeout) => {
        currentDrawTimeout = timeout;
      },
    });
  };

  const clear = () => {
    clearCanvas(ctx, canvas.width, canvas.height);
  };

  const cleanup = () => {
    if (currentDrawTimeout) {
      cancelAnimationFrame(currentDrawTimeout);
      currentDrawTimeout = null;
    }
  };

  return {
    draw,
    clear,
    cleanup,
  };
};
