import React from 'react';

const ZONE_LABELS = {
  1: 'High-Inside',   2: 'High-Middle',    3: 'High-Outside',
  4: 'Mid-Inside',    5: 'Mid-Middle',     6: 'Mid-Outside',
  7: 'Low-Inside',    8: 'Low-Middle',     9: 'Low-Outside',
  11: 'Inside (Ball)', 12: 'Above (Ball)',
  13: 'Outside (Ball)', 14: 'Below (Ball)'
};

export default function StrikeZone({ highlightedZones = [] }) {
  // Zones 1–9 form a 3×3 grid inside the strike zone
  // Zones 11–14 surround it
  const gridX = 60, gridY = 60, cellW = 56, cellH = 56;
  const gridW = cellW * 3, gridH = cellH * 3;

  const getZoneOpacity = (zone) => {
    const entry = highlightedZones.find(h => h.zone === zone);
    if (!entry) return 0;
    return Math.min(entry.probability / 15, 1); // scale for visibility
  };

  const getZoneLabel = (zone) => {
    const entry = highlightedZones.find(h => h.zone === zone);
    if (!entry) return '';
    return `${entry.probability.toFixed(1)}%`;
  };

  // Strike zone grid cells (1-9)
  const gridCells = [];
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const zone = row * 3 + col + 1;
      const x = gridX + col * cellW;
      const y = gridY + row * cellH;
      const opacity = getZoneOpacity(zone);
      gridCells.push(
        <g key={zone}>
          <rect
            x={x} y={y} width={cellW} height={cellH}
            fill={opacity > 0 ? `rgba(59, 130, 246, ${opacity})` : 'transparent'}
            stroke="#4a5a7a" strokeWidth="1"
          />
          <text
            x={x + cellW / 2} y={y + cellH / 2 - 6}
            textAnchor="middle" dominantBaseline="central"
            fill="#94a3b8" fontSize="10" fontWeight="600"
          >
            {zone}
          </text>
          <text
            x={x + cellW / 2} y={y + cellH / 2 + 8}
            textAnchor="middle" dominantBaseline="central"
            fill="#cbd5e1" fontSize="9"
          >
            {ZONE_LABELS[zone]}
          </text>
          {opacity > 0 && (
            <text
              x={x + cellW / 2} y={y + cellH / 2 + 22}
              textAnchor="middle" dominantBaseline="central"
              fill="#fbbf24" fontSize="10" fontWeight="700"
            >
              {getZoneLabel(zone)}
            </text>
          )}
        </g>
      );
    }
  }

  // Outside zones (11-14)
  const outsideZones = [
    { zone: 11, x: 0,  y: gridY,          w: gridX - 4,       h: gridH,  label: '11\nInside' },
    { zone: 12, x: gridX, y: 0,           w: gridW,            h: gridY - 4, label: '12\nAbove' },
    { zone: 13, x: gridX + gridW + 4, y: gridY, w: gridX - 4, h: gridH,  label: '13\nOutside' },
    { zone: 14, x: gridX, y: gridY + gridH + 4, w: gridW,     h: gridY - 4, label: '14\nBelow' },
  ];

  return (
    <div className="zone-container">
      <svg
        viewBox="-5 -5 290 300"
        width="280"
        height="300"
        style={{ overflow: 'visible' }}
      >
        {/* Outer strike zone border */}
        <rect
          x={gridX} y={gridY} width={gridW} height={gridH}
          fill="none" stroke="#64748b" strokeWidth="2.5" rx="2"
        />

        {/* Grid cells */}
        {gridCells}

        {/* Outside zones */}
        {outsideZones.map(oz => {
          const opacity = getZoneOpacity(oz.zone);
          return (
            <g key={oz.zone}>
              <rect
                x={oz.x} y={oz.y} width={oz.w} height={oz.h}
                fill={opacity > 0 ? `rgba(239, 68, 68, ${opacity * 0.7})` : 'rgba(30, 41, 59, 0.5)'}
                stroke="#2d3a52" strokeWidth="1" rx="4"
                strokeDasharray={opacity > 0 ? "none" : "4 2"}
              />
              <text
                x={oz.x + oz.w / 2} y={oz.y + oz.h / 2 - 4}
                textAnchor="middle" dominantBaseline="central"
                fill="#64748b" fontSize="9" fontWeight="600"
              >
                {oz.zone}
              </text>
              <text
                x={oz.x + oz.w / 2} y={oz.y + oz.h / 2 + 8}
                textAnchor="middle" dominantBaseline="central"
                fill="#64748b" fontSize="8"
              >
                {ZONE_LABELS[oz.zone]}
              </text>
              {opacity > 0 && (
                <text
                  x={oz.x + oz.w / 2} y={oz.y + oz.h / 2 + 20}
                  textAnchor="middle" dominantBaseline="central"
                  fill="#fbbf24" fontSize="10" fontWeight="700"
                >
                  {getZoneLabel(oz.zone)}
                </text>
              )}
            </g>
          );
        })}

        {/* Home plate */}
        <polygon
          points={`${gridX + gridW / 2 - 20},${gridY + gridH + 45}
                   ${gridX + gridW / 2 - 10},${gridY + gridH + 55}
                   ${gridX + gridW / 2},${gridY + gridH + 60}
                   ${gridX + gridW / 2 + 10},${gridY + gridH + 55}
                   ${gridX + gridW / 2 + 20},${gridY + gridH + 45}`}
          fill="none" stroke="#64748b" strokeWidth="1.5"
        />
        <text
          x={gridX + gridW / 2} y={gridY + gridH + 73}
          textAnchor="middle" fill="#64748b" fontSize="9"
        >
          Home Plate
        </text>
      </svg>
      <p className="zone-legend">
        Zones 1–9: Inside strike zone &nbsp;|&nbsp; Zones 11–14: Outside zone (ball)
      </p>
    </div>
  );
}
