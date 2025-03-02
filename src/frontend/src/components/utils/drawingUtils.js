// === DATA VALIDATION AND PROCESSING ===

// Utility function to validate PSD data
export const validatePSDData = (psd) => {
  return Array.isArray(psd) && psd.length === 2048;
};

// Helper function to downsample PSD data
export const downsamplePsd = (psd, targetWidth) => {
  const step = Math.floor(psd.length / targetWidth);
  const downsampled = new Array(targetWidth);

  for (let i = 0; i < targetWidth; i++) {
    let sum = 0;
    const startIdx = i * step;
    const endIdx = Math.min(startIdx + step, psd.length);

    for (let j = startIdx; j < endIdx; j++) {
      sum += psd[j];
    }

    downsampled[i] = sum / (endIdx - startIdx);
  }

  return downsampled;
};

// Helper function to scale detection indices to match downsampled data
export const scaleDetections = (detections, originalLength, targetWidth) => {
  // If both detection values are 0, return [0, 0]
  if (detections[0] === 0 && detections[1] === 0) {
    return [0, 0];
  }

  const scale = targetWidth / originalLength;
  return [
    Math.floor(detections[0] * scale),
    Math.floor(detections[1] * scale)
  ];
};

// === IMAGE AND CANVAS OPERATIONS ===

export const createRowImageData = (
  width,
  downsampled,
  colorMap,
  scaledDetections,
  minDbm,
  maxDbm,
) => {
  // Create a 2-pixel high row
  const imgData = new ImageData(width, 2);
  const data = imgData.data;
  const dbmRange = maxDbm - minDbm;
  const pointsToProcess = width / 2;
  const [detStart, detEnd] = scaledDetections;

  for (let i = 0; i < pointsToProcess; i++) {
    // Calculate normalized value more efficiently
    const normalized = Math.max(0, Math.min(1, (downsampled[i] - minDbm) / dbmRange));
    const colorIdx = Math.min(255, Math.floor(normalized * 255));

    // Get color once
    const [r, g, b, a] = colorMap[colorIdx] || [0, 0, 0, 255];
    const alpha = (detStart < i && detEnd > i) ? a : a / 2;

    // Calculate base indices for the 2x2 pixel block
    const baseIdx1 = (i * 2) * 4;
    const baseIdx2 = baseIdx1 + 4;
    const baseIdx3 = width * 4 + baseIdx1;
    const baseIdx4 = width * 4 + baseIdx2;

    // Set all 4 pixels more efficiently
    // Top left pixel
    data[baseIdx1] = r;
    data[baseIdx1 + 1] = g;
    data[baseIdx1 + 2] = b;
    data[baseIdx1 + 3] = alpha;

    // Top right pixel
    data[baseIdx2] = r;
    data[baseIdx2 + 1] = g;
    data[baseIdx2 + 2] = b;
    data[baseIdx2 + 3] = alpha;

    // Bottom left pixel
    data[baseIdx3] = r;
    data[baseIdx3 + 1] = g;
    data[baseIdx3 + 2] = b;
    data[baseIdx3 + 3] = alpha;

    // Bottom right pixel
    data[baseIdx4] = r;
    data[baseIdx4 + 1] = g;
    data[baseIdx4 + 2] = b;
    data[baseIdx4 + 3] = alpha;
  }

  return imgData;
};

// Utility function to shift image data up
export const shiftImageUp = (ctx, width, height) => {
  try {
    const image = ctx.getImageData(0, 2, width, height - 2);
    ctx.putImageData(image, 0, 0);
    return true;
  } catch (error) {
    console.error("Error shifting image data:", error);
    return false;
  }
};

export const clearCanvas = (ctx, width, height) => {
  ctx.clearRect(0, 0, width, height);
};

// === MAIN DRAWING FUNCTIONS ===

// Main drawing function
export const drawSpectrogram = ({
  canvas,
  psd,
  colorMap,
  detections,
  minDbm,
  maxDbm,
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

      // Check if previous draw is still in progress
      if (drawTimeout) {
        resolve();
        return;
      }

      // Setup canvas context
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      const width = canvas.width;
      const height = canvas.height;

      // Schedule drawing
      setDrawTimeout(
        requestAnimationFrame(() => {
          // Downsample PSD data to half the canvas width (since we're using 2x2 pixels)
          const pointsToDisplay = Math.floor(width / 2);
          const downsampled = downsamplePsd(psd, pointsToDisplay);
          const scaledDetections = scaleDetections(detections, psd.length, pointsToDisplay);
          const imgData = createRowImageData(
            width,
            downsampled,
            colorMap,
            scaledDetections,
            minDbm,
            maxDbm,
          );

          // Shift existing image up
          if (!shiftImageUp(ctx, width, height)) {
            onError("Error shifting image data");
          }

          // Draw new row at the bottom
          ctx.putImageData(imgData, 0, height - 2);

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

export const createSpectrogramDrawer = (canvas, colorMap, onError) => {
  let currentDrawTimeout = null;
  let psdHistory = [];
  let detectionHistory = [];
  let lastRangeDbm = [-110, -70]; // Default range
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const maxHistorySize = canvas.height / 2; // Each row is 2px high

  const redrawFullCanvas = (rangeDbm) => {
    clearCanvas(ctx, canvas.width, canvas.height);

    // Process each row of history from oldest to newest
    for (let i = 0; i < psdHistory.length; i++) {
      const psd = psdHistory[i];
      const detections = detectionHistory[i];

      // Calculate position from bottom (newest data at bottom)
      const y = canvas.height - ((psdHistory.length - i) * 2);
      if (y < 0) continue; // Skip if it would be drawn outside the canvas

      // Downsample and create row image data
      const pointsToDisplay = Math.floor(canvas.width / 2);
      const downsampled = downsamplePsd(psd, pointsToDisplay);
      const scaledDetections = scaleDetections(detections, psd.length, pointsToDisplay);
      const imgData = createRowImageData(
        canvas.width,
        downsampled,
        colorMap,
        scaledDetections,
        rangeDbm[0],
        rangeDbm[1]
      );

      // Draw directly at the calculated position
      ctx.putImageData(imgData, 0, y);
    }
  };

  const draw = (psd, rangeDbm, detections) => {
    // Update histories
    psdHistory.push(psd);
    detectionHistory.push(detections);
    if (psdHistory.length > maxHistorySize) {
      psdHistory.shift();
      detectionHistory.shift();
    }

    lastRangeDbm = rangeDbm;

    return drawSpectrogram({
      canvas,
      psd,
      colorMap,
      detections,
      minDbm: rangeDbm[0],
      maxDbm: rangeDbm[1],
      onError,
      drawTimeout: currentDrawTimeout,
      setDrawTimeout: (timeout) => {
        currentDrawTimeout = timeout;
      },
    });
  };

  const updateRange = (rangeDbm) => {
    // Cancel any pending draw
    if (currentDrawTimeout) {
      cancelAnimationFrame(currentDrawTimeout);
      currentDrawTimeout = null;
    }

    lastRangeDbm = rangeDbm;

    // Immediately redraw the entire canvas with the new range
    if (psdHistory.length > 0) {
      redrawFullCanvas(rangeDbm);
    }

    return Promise.resolve();
  };

  const clear = () => {
    clearCanvas(ctx, canvas.width, canvas.height);
    psdHistory = [];
    detectionHistory = [];
  };

  const cleanup = () => {
    if (currentDrawTimeout) {
      cancelAnimationFrame(currentDrawTimeout);
      currentDrawTimeout = null;
    }
    psdHistory = [];
    detectionHistory = [];
  };

  return {
    draw,
    updateRange,
    clear,
    cleanup,
  };
};
