import React, { useRef, useState } from 'react';

interface ButtonProps {
  color: string,
  onClick: () => void;
  disabled?: boolean;
  children?: React.ReactNode;
  className?: string;
}

const Button: React.FC<ButtonProps> = ({ 
  color,
  onClick, 
  disabled = false, 
  children = "Clear",
  className = ""
}) => {
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);

  const handleMouseMove = (e: React.MouseEvent<HTMLButtonElement>) => {
    if (buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      setMousePosition({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      });
    }
  };

  const handleMouseEnter = () => {
    setIsHovering(true);
  };

  const handleMouseLeave = () => {
    setIsHovering(false);
  };

  return (
    <button
      ref={buttonRef}
      color={color}
      onClick={onClick}
      disabled={disabled}
      onMouseMove={handleMouseMove}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={`
        relative overflow-hidden
        px-6 py-3 
        text-white font-semibold
        rounded-xl
        shadow-lg hover:shadow-xl
        transform hover:scale-105
        transition-all duration-200 ease-out
        disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
        focus:outline-none focus:ring-0
        outline-none
        ${className}
      `}
      style={{
        background: disabled ? '#6b7280' : (
          isHovering ? `
            radial-gradient(circle 100px at ${mousePosition.x}px ${mousePosition.y}px, 
              rgba(255,255,255,0.2) 0%, 
              rgba(255,255,255,0.1) 40%, 
              transparent 70%),
            ${color}
          ` : color
        )
      }}
    >
      <span className="relative z-10">
        {children}
      </span>
    </button>
  );
};

export default Button;