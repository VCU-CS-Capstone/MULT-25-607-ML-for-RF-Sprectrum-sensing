// drawingUtils.js

import { downsamplePSD, scaleDetections, isValidPSD } from "./psdUtils";

export const DetectionType = {
  NONE: 0,
  WIFI: 1,
  BLUETOOTH: 2,
};

export class SpectrogramDrawer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {Array} colorMap
   * @param {Function} onError
   * @param {number} pixelSize
   */
  constructor(canvas, colorMap, onError = () => {}, pixelSize = 8) {
    this.canvas = canvas;
    this.colorMap = colorMap;
    this.onError = onError;
    this.pixelSize = pixelSize;

    this.ctx = canvas.getContext("2d", { willReadFrequently: true });
    this.psdHistory = [];
    this.detectionsHistory = [];
    this.detectionTypeHistory = [];
    this.lastRangeDbm = [-110, -70];
    this.currentDrawTimeout = null;
    this.maxHistorySize = Math.floor(canvas.height / pixelSize);
  }

  /**
   * Create a single row of spectrogram image data.
   */
  createSpectrogramRow(psd, detections, minDbm, maxDbm, detectionType) {
    const width = this.canvas.width;
    const pixelSize = this.pixelSize;
    const points = Math.floor(width / pixelSize);
    const downsampled = downsamplePSD(psd, points);
    const [detStart, detEnd] = scaleDetections(detections, psd.length, points);

    const rowHeight = pixelSize;
    const imgData = new ImageData(width, rowHeight);
    const data = imgData.data;
    const dbmRange = maxDbm - minDbm;

    for (let i = 0; i < points; i++) {
      const normalized = Math.max(
        0,
        Math.min(1, (downsampled[i] - minDbm) / dbmRange),
      );
      const colorIdx = Math.min(255, Math.floor(normalized * 255));
      let r, g, b, a;

      // Highlight detection region
      if (detectionType === DetectionType.WIFI && detStart < i && detEnd > i) {
        [r, g, b, a] = [0, 255, 0, 255];
      } else if (
        detectionType === DetectionType.BLUETOOTH &&
        detStart < i &&
        detEnd > i
      ) {
        [r, g, b, a] = [255, 0, 0, 255];
      } else {
        [r, g, b, a] = this.colorMap[colorIdx] || [0, 0, 0, 255];
      }

      const alpha = detStart < i && detEnd > i ? a : a / 2;

      // Fill the pixel block
      for (let x = 0; x < pixelSize; x++) {
        for (let y = 0; y < rowHeight; y++) {
          const baseIdx = (y * width + i * pixelSize + x) * 4;
          data[baseIdx] = r;
          data[baseIdx + 1] = g;
          data[baseIdx + 2] = b;
          data[baseIdx + 3] = alpha;
        }
      }
    }

    return imgData;
  }

  /**
   * Draws a new row on the spectrogram.
   */
  draw(psd, rangeDbm, detections, detectionType) {
    return new Promise((resolve, reject) => {
      try {
        if (!this.canvas) throw new Error("Canvas not provided");
        if (!isValidPSD(psd)) throw new Error("Invalid PSD data format");
        if (this.currentDrawTimeout) return resolve();

        // Update histories
        this.psdHistory.push(psd);
        this.detectionsHistory.push(detections);
        this.detectionTypeHistory.push(detectionType);
        if (this.psdHistory.length > this.maxHistorySize) {
          this.psdHistory.shift();
          this.detectionsHistory.shift();
          this.detectionTypeHistory.shift();
        }
        this.lastRangeDbm = rangeDbm;

        this.currentDrawTimeout = requestAnimationFrame(() => {
          const imgData = this.createSpectrogramRow(
            psd,
            detections,
            rangeDbm[0],
            rangeDbm[1],
            detectionType,
          );

          if (
            !SpectrogramDrawer.shiftCanvasUp(
              this.ctx,
              this.canvas.width,
              this.canvas.height,
              this.pixelSize,
            )
          ) {
            this.onError("Error shifting image data");
          }

          this.ctx.putImageData(
            imgData,
            0,
            this.canvas.height - this.pixelSize,
          );
          this.currentDrawTimeout = null;
          resolve();
        });
      } catch (error) {
        console.error("Error in draw:", error);
        this.onError("Error drawing spectrogram.");
        reject(error);
      }
    });
  }

  /**
   * Redraws the entire canvas with the current history and range.
   */
  redrawFullCanvas(rangeDbm) {
    SpectrogramDrawer.clearCanvas(
      this.ctx,
      this.canvas.width,
      this.canvas.height,
    );

    for (let i = 0; i < this.psdHistory.length; i++) {
      const psd = this.psdHistory[i];
      const detections = this.detectionsHistory[i];
      const detectionType = this.detectionTypeHistory[i];
      const y =
        this.canvas.height - (this.psdHistory.length - i) * this.pixelSize;
      if (y < 0) continue;

      const imgData = this.createSpectrogramRow(
        psd,
        detections,
        rangeDbm[0],
        rangeDbm[1],
        detectionType,
      );
      this.ctx.putImageData(imgData, 0, y);
    }
  }

  /**
   * Updates the dBm range and redraws the canvas.
   */
  updateRange(rangeDbm) {
    if (this.currentDrawTimeout) {
      cancelAnimationFrame(this.currentDrawTimeout);
      this.currentDrawTimeout = null;
    }
    this.lastRangeDbm = rangeDbm;
    if (this.psdHistory.length > 0) {
      this.redrawFullCanvas(rangeDbm);
    }
    return Promise.resolve();
  }

  /**
   * Clears the canvas and all history.
   */
  clear() {
    SpectrogramDrawer.clearCanvas(
      this.ctx,
      this.canvas.width,
      this.canvas.height,
    );
    this.psdHistory = [];
    this.detectionsHistory = [];
    this.detectionTypeHistory = [];
  }

  /**
   * Cleans up any pending draws and clears history.
   */
  cleanup() {
    if (this.currentDrawTimeout) {
      cancelAnimationFrame(this.currentDrawTimeout);
      this.currentDrawTimeout = null;
    }
    this.psdHistory = [];
    this.detectionsHistory = [];
    this.detectionTypeHistory = [];
  }

  // --- Static helpers ---

  static clearCanvas(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);
  }

  static shiftCanvasUp(ctx, width, height, pixelSize) {
    try {
      const image = ctx.getImageData(0, pixelSize, width, height - pixelSize);
      ctx.putImageData(image, 0, 0);
      return true;
    } catch (error) {
      console.error("Error shifting image data:", error);
      return false;
    }
  }
}
