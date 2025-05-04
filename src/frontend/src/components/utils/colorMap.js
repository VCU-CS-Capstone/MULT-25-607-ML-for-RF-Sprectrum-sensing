// colorMap.js

export function generateColorMap() {
  const colMap = [];
  for (let i = 0; i < 256; i++) {
    let r, g, b;
    if (i < 64) {
      r = 0;
      g = i * 4;
      b = 255;
    } else if (i < 128) {
      r = 0;
      g = 255;
      b = 255 - (i - 64) * 4;
    } else if (i < 192) {
      r = (i - 128) * 4;
      g = 255;
      b = 0;
    } else {
      r = 255;
      g = 255 - (i - 192) * 4;
      b = 0;
    }
    colMap.push([r, g, b, 255]);
  }
  return colMap;
}
