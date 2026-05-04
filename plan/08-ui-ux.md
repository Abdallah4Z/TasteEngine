# UI/UX Design Plan

## Overview

Professional, clean web interface with glassmorphism aesthetics. Three main pages with smooth navigation, interactive elements, and visual data presentation.

## Design System

| Element | Specification |
|---------|---------------|
| **Style** | Glassmorphism + Minimalist |
| **Primary Color** | #6C63FF (purple-blue) |
| **Secondary** | #FF6584 (coral accent) |
| **Background** | Gradient dark (#0f0c29 → #302b63 → #24243e) |
| **Glass panels** | backdrop-filter: blur(20px), semi-transparent bg |
| **Font** | Inter (Google Fonts) |
| **Cards** | Rounded (16px), subtle shadows, hover animations |

## Page 1: Home / User Selection (`index.html`)

**Layout**:
```
┌─────────────────────────────────────────────┐
│  ⭐ Intelligent Recommender System    [nav] │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────── User Selection ────────────┐  │
│  │  Select User: [Dropdown ▼]            │  │
│  │  ┌────────────────────────────────┐   │  │
│  │  │ User Profile Card              │   │  │
│  │  │ Name: Alice | Age: 28         │   │  │
│  │  │ Prefs: Electronics, Books     │   │  │
│  │  │ Budget: $100 - $1500          │   │  │
│  │  └────────────────────────────────┘   │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  ┌───── Select Approach ─────────────────┐  │
│  │  [🤝 Collaborative]  [🏷️ Content] [⚙️ Knowledge] │
│  │  [🔁 Compare All]                       │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  [🚀 Generate Recommendations]              │
└─────────────────────────────────────────────┘
```

**Interactions**:
- Dropdown updates profile card in real-time
- Approach buttons are toggle-style with highlighting
- Hover effects on all interactive elements

## Page 2: Recommendations (`recommend.html`)

**Layout**:
```
┌─────────────────────────────────────────────┐
│  ⭐ Results for Alice              [nav]   │
├─────────────────────────────────────────────┤
│  ┌─ Approach Tabs ───────────────────────┐  │
│  │  [CF] [Content] [Knowledge] [Compare] │  │
│  └────────────────────────────────────────┘  │
│                                             │
│  ┌─ Recommendations ──────────────────────┐ │
│  │  ┌─────────────────────────────────┐   │ │
│  │  │ 🖼️ Samsung Galaxy S24        │   │ │
│  │  │ ⭐⭐⭐⭐☆ 4.2  |  $799.99     │   │ │
│  │  │ 💡 12 similar users purchased │   │ │
│  │  └─────────────────────────────────┘   │ │
│  │  ┌─────────────────────────────────┐   │ │
│  │  │ 🖼️ Sony WH-1000XM5            │   │ │
│  │  │ ⭐⭐⭐⭐⭐ 4.8  |  $349.99    │   │ │
│  │  │ 💡 91% similar to liked item   │   │ │
│  │  └─────────────────────────────────┘   │ │
│  │  ... (top 10 items)                    │ │
│  └─────────────────────────────────────────┘ │
│                                             │
│  [📊 View Evaluation & Comparison]           │
└─────────────────────────────────────────────┘
```

**Interactions**:
- Tab switching animates between approaches
- "Compare" tab shows side-by-side view
- Each card has hover lift effect

## Page 3: Evaluation Dashboard (`evaluation.html`)

**Layout**:
```
┌─────────────────────────────────────────────┐
│  📊 Evaluation & Comparison        [nav]   │
├─────────────────────────────────────────────┤
│  ┌─── CF Methods Comparison ─────────────┐  │
│  │  [Bar Chart: RMSE, MAE by method]    │  │
│  │  [Table: All metrics × All methods]  │  │
│  │  🏆 Best CF Method: SVD (RMSE: 0.81) │  │
│  └────────────────────────────────────────┘  │
│                                             │
│  ┌─── Approach Comparison ────────────────┐ │
│  │  [Grouped Bar Chart: CF vs CB vs KB]  │ │
│  │  [Radar Chart: Strengths profile]     │ │
│  │  🏆 Best Approach: CF (overall)       │ │
│  └────────────────────────────────────────┘  │
│                                             │
│  ┌─── Condition Analysis ─────────────────┐ │
│  │  [Table: Best approach per condition] │ │
│  │  📝 Analysis: Why differences occur   │ │
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**Charts** (using Chart.js):
- Grouped bar chart for CF methods comparison
- Horizontal bar chart for approach comparison
- Radar chart for strength profiles
- Condition matrix table

## Responsive Design

- Mobile-friendly card layouts
- Collapsible navigation
- Touch-friendly buttons
- Smooth CSS transitions (300ms ease)

## Files

| File | Path |
|------|------|
| Main CSS | `static/css/style.css` |
| Home template | `templates/index.html` |
| Recommend template | `templates/recommend.html` |
| Evaluation template | `templates/evaluation.html` |
