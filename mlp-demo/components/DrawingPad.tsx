'use client';

import React, { useState, useRef, useCallback, JSX } from 'react';

export type PixelUpdate = {
  row: number;
  col: number;
  value: number;
}

export type PixelGrid = number[][];

export type Prediction = {
  class: string | number;
  confidence: number;
}

// Define props interface
interface DrawingPadProps {
  pixels: PixelGrid;
  onPixelUpdate: (updates: PixelUpdate[]) => void;
  className?: string;
}

const DrawingPad: React.FC<DrawingPadProps> = ({ pixels, onPixelUpdate, className = "" }) => {
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const [blurAmount, setBlurAmount] = useState<number>(0.5);
  const padRef = useRef<HTMLDivElement>(null);

  const drawPixel = useCallback((row: number, col: number, intensity: number = 1) => {
    const radius = 4;
    const updates: PixelUpdate[] = [];
    
    for (let i = -radius; i <= radius; i++) {
      for (let j = -radius; j <= radius; j++) {
        const newRow = row + i;
        const newCol = col + j;
        
        if (newRow >= 0 && newRow < 28 && newCol >= 0 && newCol < 28) {
          const distance = Math.sqrt(i*i + j*j);
          let value = intensity;
          
          if (distance > radius/1.5) {
            value = intensity * (1 - (distance - radius/1.5) / (radius - radius/1.5));
          }
          
          value = Math.max(0, Math.min(1, value));
          
          if (value > pixels[newRow][newCol]) {
            updates.push({ row: newRow, col: newCol, value });
          }
        }
      }
    }
    
    if (updates.length > 0) {
      onPixelUpdate(updates);
    }
  }, [pixels, onPixelUpdate]);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    setIsDrawing(true);
    const target = e.target as HTMLDivElement;
    if (target.dataset.row && target.dataset.col) {
      const row = parseInt(target.dataset.row);
      const col = parseInt(target.dataset.col);
      drawPixel(row, col);
    }
  }, [drawPixel]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDrawing) return;
    
    const target = e.target as HTMLDivElement;
    if (target.dataset.row && target.dataset.col) {
      const row = parseInt(target.dataset.row);
      const col = parseInt(target.dataset.col);
      drawPixel(row, col);
    }
  }, [isDrawing, drawPixel]);

  const handleMouseUp = useCallback(() => {
    setIsDrawing(false);
  }, []);

  const handleTouchStart = useCallback((e: React.TouchEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDrawing(true);
    const touch = e.touches[0];
    const target = document.elementFromPoint(touch.clientX, touch.clientY) as HTMLDivElement;
    if (target && target.dataset.row && target.dataset.col) {
      const row = parseInt(target.dataset.row);
      const col = parseInt(target.dataset.col);
      drawPixel(row, col);
    }
  }, [drawPixel]);

  const handleTouchMove = useCallback((e: React.TouchEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!isDrawing) return;
    
    const touch = e.touches[0];
    const target = document.elementFromPoint(touch.clientX, touch.clientY) as HTMLDivElement;
    if (target && target.dataset.row && target.dataset.col) {
      const row = parseInt(target.dataset.row);
      const col = parseInt(target.dataset.col);
      drawPixel(row, col);
    }
  }, [isDrawing, drawPixel]);

  const renderPixel = (row: number, col: number): JSX.Element => {
    const value = pixels[row][col];
    const colorValue = Math.floor(value * 255);
    const backgroundColor = `rgb(${colorValue}, ${colorValue}, ${colorValue})`;
    
    return (
      <div
        key={`${row}-${col}`}
        className="cursor-crosshair aspect-square"
        style={{ backgroundColor }}
        data-row={row}
        data-col={col}
      />
    );
  };

  return (
    <div 
      ref={padRef}
      className={`grid grid-cols-28 gap-0 border-2 border-gray-600 select-none ${className}`}
      style={{ 
        gridTemplateColumns: 'repeat(28, 1fr)',
        touchAction: 'none'
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleMouseUp}
    >
      {Array(28).fill(null).map((_, row) =>
        Array(28).fill(null).map((_, col) => renderPixel(row, col))
      )}
    </div>
  );
};

export default DrawingPad;