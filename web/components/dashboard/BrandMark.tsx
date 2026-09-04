"use client";

import { useId } from "react";

import { cn } from "@/lib/utils";

/**
 * The Sports Edge mark: a baseball cut through on a 32° plane, the halves slid
 * apart along it, the lower one turned to show its cut face — leather cover,
 * wool windings, rubber core.
 *
 * Inlined rather than loaded as a file so it stays crisp and costs no request.
 * The gradients, filters and clips are namespaced with `useId` because the mark
 * renders more than once per page (header and mobile sheet) and duplicate def
 * ids would cross-wire the two.
 *
 * Keep in sync with `app/icon.svg`, which is the same artwork as a standalone
 * file — Next's icon convention needs a real file at that path.
 */
export function BrandMark({ className }: { className?: string }) {
  const uid = useId().replace(/:/g, "");
  const id = (name: string) => `${uid}-${name}`;
  const url = (name: string) => `url(#${id(name)})`;

  return (
    <svg
      viewBox="0 0 128 128"
      className={cn("size-7 shrink-0", className)}
      role="img"
      aria-label="Sports Edge"
    >
      <defs>
        <linearGradient id={id("fld")} x1="0" y1="0" x2="0.35" y2="1">
          <stop offset="0" stopColor="#1B2E7A" />
          <stop offset="1" stopColor="#070C24" />
        </linearGradient>
        <radialGradient id={id("lit")} cx="0.26" cy="0.18" r="0.9">
          <stop offset="0" stopColor="#fff" stopOpacity="0.22" />
          <stop offset="1" stopColor="#fff" stopOpacity="0" />
        </radialGradient>
        <radialGradient id={id("lea")} cx="0.3" cy="0.24" r="0.92">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="0.45" stopColor="#f7f6f2" />
          <stop offset="0.84" stopColor="#d2d3da" />
          <stop offset="1" stopColor="#a4a6b1" />
        </radialGradient>
        <radialGradient id={id("dom")} cx="0.62" cy="0.74" r="0.85">
          <stop offset="0" stopColor="#e9e8e3" />
          <stop offset="1" stopColor="#8d8f99" />
        </radialGradient>
        <radialGradient id={id("yrn")} cx="0.5" cy="0.44" r="0.62">
          <stop offset="0" stopColor="#f4ecd8" />
          <stop offset="0.7" stopColor="#e4d6b4" />
          <stop offset="1" stopColor="#c2ae86" />
        </radialGradient>
        <radialGradient id={id("cor")} cx="0.38" cy="0.34" r="0.8">
          <stop offset="0" stopColor="#e8635a" />
          <stop offset="1" stopColor="#a92b25" />
        </radialGradient>
        <linearGradient id={id("bld")} x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#ffffff" stopOpacity="0" />
          <stop offset="0.35" stopColor="#ffffff" stopOpacity="1" />
          <stop offset="1" stopColor="#ffffff" stopOpacity="0.15" />
        </linearGradient>
        <filter id={id("glo")} x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="2.2" />
        </filter>
        <filter id={id("shd")} x="-40%" y="-40%" width="180%" height="180%">
          <feDropShadow dx="1.5" dy="3" stdDeviation="3" floodColor="#000" floodOpacity="0.45" />
        </filter>
        <clipPath id={id("bdg")}>
          <rect width="128" height="128" rx="28" />
        </clipPath>
        <clipPath id={id("hiL")}>
          <rect x="-300" y="-300" width="760" height="358" transform="rotate(-32 55 58)" />
        </clipPath>
        <clipPath id={id("hiR")}>
          <rect x="-300" y="70" width="760" height="400" transform="rotate(-32 73 70)" />
        </clipPath>
      </defs>

      <g clipPath={url("bdg")}>
        <rect width="128" height="128" fill={url("fld")} />
        <rect width="128" height="128" fill={url("lit")} />

        {/* Lower half: dome behind, cut face on the plane */}
        <g filter={url("shd")}>
          <g clipPath={url("hiR")}>
            <circle cx="73" cy="70" r="38" fill={url("dom")} />
          </g>
          <g transform="rotate(-32 73 70)">
            <ellipse cx="73" cy="70" rx="38" ry="15.5" fill="#eceae4" />
            <ellipse cx="73" cy="70" rx="35.4" ry="13.4" fill={url("yrn")} />
            <ellipse cx="73" cy="70" rx="13" ry="5.1" fill={url("cor")} />
            <ellipse cx="64" cy="66.6" rx="12" ry="3.4" fill="#fff" opacity="0.3" />
          </g>
        </g>

        {/* Upper half: leather, stitched seam, specular */}
        <g clipPath={url("hiL")} filter={url("shd")}>
          <circle cx="55" cy="58" r="38" fill={url("lea")} />
          <g fill="none" stroke="#c8342c" strokeLinecap="butt">
            <path d="M26,64 C30,36 54,24 82,32" opacity="0.5" strokeWidth="1.8" />
            <path d="M26,64 C30,36 54,24 82,32" strokeWidth="6.5" strokeDasharray="2 4.8" />
          </g>
          <line x1="22.8" y1="78.1" x2="87.2" y2="37.9" stroke="#5a5c66" strokeOpacity="0.35" strokeWidth="4" />
          <ellipse cx="42" cy="39" rx="15" ry="9" fill="#fff" opacity="0.55" transform="rotate(-38 42 39)" />
        </g>

        {/* Blade through the gap */}
        <g filter={url("glo")} opacity="0.55">
          <path d="M-16,116 Q61,58 144,12 Q66,63 -16,116 Z" fill="#8FA8FF" />
        </g>
        <path d="M-16,116 Q61,58 144,12 Q66,63 -16,116 Z" fill={url("bld")} />
        <path d="M-10,110 Q62,60 138,16" fill="none" stroke="#ffffff" strokeWidth="2" opacity="0.95" />

        <g fill="#ffffff" opacity="0.85">
          <circle cx="104" cy="34" r="2.4" />
          <circle cx="114" cy="46" r="1.6" />
          <circle cx="26" cy="96" r="2" />
        </g>
        <rect width="128" height="128" rx="28" fill="none" stroke="#fff" strokeOpacity="0.14" strokeWidth="2" />
      </g>
    </svg>
  );
}
