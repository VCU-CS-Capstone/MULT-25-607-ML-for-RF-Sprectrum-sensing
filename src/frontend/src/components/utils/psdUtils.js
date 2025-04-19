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

