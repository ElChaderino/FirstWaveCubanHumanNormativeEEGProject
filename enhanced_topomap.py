#!/usr/bin/env python3
"""
Enhanced Topographical Map Module for EEG Clinical Analysis
CLINICAL GRADE VERSION — Matches NeuroGuide/ClinicalQ standards
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.interpolate import griddata
from scipy.spatial.distance import pdist, squareform
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== CLINICAL CONFIG ==========

DEFAULT_CMAP = 'RdBu_r'
DEFAULT_VLIM = (-3, 3)
MONTAGE_NAME = "standard_1020"

# ===== EEG PARADOX THEME =====
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

CYBER_BG      = '#0a0f17'   # deep ink
CYBER_FG      = '#d6f6ff'   # pale neon text
NEON_CYAN     = '#00e5ff'
NEON_MAGENTA  = '#ff2bd6'
NEON_LIME     = '#b7ff00'
NEON_YELLOW   = '#ffe600'
NEON_ORANGE   = '#ff8a00'
NEON_RED      = '#ff3355'
NEON_PURPLE   = '#9b5cff'

def _make_paradox_div():  # for z-scores (diverging)
    return LinearSegmentedColormap.from_list(
        'paradox_div',
        [NEON_CYAN, '#2b9df7', '#1e4fa3', '#171b2c', '#7a184f', NEON_MAGENTA, NEON_ORANGE],
        N=256
    )

def _make_paradox_seq():  # for power (sequential)
    return LinearSegmentedColormap.from_list(
        'paradox_seq',
        ['#162133', '#173a5e', '#1c6aa7', NEON_CYAN, NEON_LIME, NEON_YELLOW, NEON_ORANGE],
        N=256
    )

PARADOX_CMAP_DIV = _make_paradox_div()
PARADOX_CMAP_SEQ = _make_paradox_seq()

def _halo(**kw):
    """White outer glow for text/lines on dark bg."""
    lw = kw.get('lw', 3.5); color = kw.get('color', 'black'); alpha = kw.get('alpha', 0.9)
    return [pe.withStroke(linewidth=lw, foreground=color, alpha=alpha)]

def _neon_glow_circle(ax, center=(0,0), radius=1.0, color=NEON_CYAN):
    # multi-stroke glow
    for w, a in [(8,0.10),(5,0.18),(3,0.35)]:
        c = plt.Circle(center, radius, fill=False, color=color, linewidth=w, alpha=a, zorder=2)
        ax.add_patch(c)
    ax.add_patch(plt.Circle(center, radius, fill=False, color='#ffffff', linewidth=1.6, zorder=3))

def _region_color(ch):
    ch = ch.upper()
    if ch.startswith('F'):  return NEON_MAGENTA
    if ch.startswith('C'):  return NEON_CYAN
    if ch.startswith('P'):  return NEON_LIME
    if ch.startswith('O'):  return NEON_YELLOW
    if ch.startswith('T'):  return NEON_PURPLE
    return CYBER_FG

def _severity_color(z):
    az = abs(float(z))
    if az >= 2.58: return NEON_RED
    if az >= 2.00: return NEON_ORANGE
    if az >= 1.50: return NEON_YELLOW
    return CYBER_FG

# Clinical-grade 10-20 positions (exact NeuroGuide coordinates)
# Using proper MNE case format to match clean_channel_name output
CLINICAL_1020_POSITIONS = {
    'Fp1': (-0.31, 0.95), 'Fpz': (0.0, 0.95), 'Fp2': (0.31, 0.95),
    'F7': (-0.71, 0.71), 'F3': (-0.45, 0.71), 'Fz': (0.0, 0.71), 
    'F4': (0.45, 0.71), 'F8': (0.71, 0.71),
    'T7': (-0.95, 0.31), 'C3': (-0.45, 0.31), 'Cz': (0.0, 0.31),
    'C4': (0.45, 0.31), 'T8': (0.95, 0.31),
    'T3': (-0.95, 0.31), 'T4': (0.95, 0.31),  # Old nomenclature
    'P7': (-0.71, -0.31), 'P3': (-0.45, -0.31), 'Pz': (0.0, -0.31),
    'P4': (0.45, -0.31), 'P8': (0.71, -0.31),
    'T5': (-0.71, -0.31), 'T6': (0.71, -0.31),  # Old nomenclature
    'O1': (-0.31, -0.95), 'Oz': (0.0, -0.95), 'O2': (0.31, -0.95),
    # Add uppercase versions for backward compatibility
    'FP1': (-0.31, 0.95), 'FPZ': (0.0, 0.95), 'FP2': (0.31, 0.95),
    'FZ': (0.0, 0.71), 'CZ': (0.0, 0.31), 'PZ': (0.0, -0.31), 'OZ': (0.0, -0.95)
}

# ========== HELPER FUNCTIONS ==========

def create_custom_1020_positions(channel_names):
    """Create custom 10-20 positions for channels"""
    # Standard 10-20 positions (x, y) - normalized to unit circle
    standard_positions = {
        'Fp1': (-0.3, 0.8), 'Fp2': (0.3, 0.8),
        'F7': (-0.8, 0.4), 'F3': (-0.4, 0.4),
        'Fz': (0.0, 0.4), 'F4': (0.4, 0.4),
        'F8': (0.8, 0.4), 'T7': (-0.8, 0.0),
        'C3': (-0.4, 0.0), 'Cz': (0.0, 0.0),
        'C4': (0.4, 0.0), 'T8': (0.8, 0.0),
        'P7': (-0.8, -0.4), 'P3': (-0.4, -0.4),
        'Pz': (0.0, -0.4), 'P4': (0.4, -0.4),
        'P8': (0.8, -0.4), 'O1': (-0.3, -0.8),
        'Oz': (0.0, -0.8), 'O2': (0.3, -0.8)
    }
    
    # Create positions for each channel
    pos_array = []
    for ch_name in channel_names:
        # Clean channel name (remove common suffixes)
        clean_name = ch_name.replace('-LE', '').replace('-RE', '').replace('-REF', '')
        clean_name = clean_name.replace('-M1', '').replace('-M2', '')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c in '-_')
        clean_name = clean_name.strip()
        
        if clean_name in standard_positions:
            pos_array.append(standard_positions[clean_name])
        else:
            # Try to find a close match
            found_match = False
            for std_name, pos in standard_positions.items():
                if std_name in clean_name or clean_name in std_name:
                    pos_array.append(pos)
                    found_match = True
                    break
            
            if not found_match:
                # Estimate position based on naming pattern
                x, y = estimate_channel_position(clean_name)
                pos_array.append((x, y))
    
    return np.array(pos_array)

def estimate_channel_position(channel_name):
    """Estimate channel position based on naming conventions"""
    # Extract letter and number from channel name
    letters = ''.join([c for c in channel_name if c.isalpha()])
    numbers = ''.join([c for c in channel_name if c.isdigit()])
    
    # Determine Y position based on letter
    if 'F' in letters.upper():
        y = 0.7
    elif 'C' in letters.upper():
        y = 0.5
    elif 'P' in letters.upper():
        y = 0.3
    elif 'O' in letters.upper():
        y = 0.1
    else:
        y = 0.5
    
    # Determine X position based on number and side
    if any(side in numbers for side in ['1', '3', '5', '7', '9']):
        x = -0.35  # Left side
    elif any(side in numbers for side in ['2', '4', '6', '8', '10']):
        x = 0.35   # Right side
    elif 'Z' in letters.upper():
        x = 0.0    # Midline
    else:
        x = 0.0
    
    return x, y

def ensure_montage(info):
    """Ensure info object has proper montage attached"""
    if not info.get_montage():
        try:
            montage = mne.channels.make_standard_montage(MONTAGE_NAME)
            info.set_montage(montage, on_missing='ignore')  # Key fix: ignore missing channels
            logger.info(f"Applied default montage: {MONTAGE_NAME}")
        except Exception as e:
            logger.warning(f"Could not apply standard montage: {e}")
            # Try alternative montages
            try:
                montage = mne.channels.make_standard_montage("biosemi64")
                info.set_montage(montage, on_missing='ignore')  # Key fix here too
                logger.info("Applied biosemi64 montage as fallback")
            except Exception as e2:
                try:
                    montage = mne.channels.make_standard_montage("standard_1005")
                    info.set_montage(montage, on_missing='ignore')  # And here
                    logger.info("Applied standard_1005 montage as fallback")
                except Exception as e3:
                    logger.error(f"All montage attempts failed: {e3}")

def get_clean_positions(info):
    """Get clean channel positions with proper montage handling"""
    ensure_montage(info)
    picks = mne.pick_types(info, eeg=True)
    
    try:
        # Try to use MNE's internal position preparation (most reliable)
        try:
            # For newer MNE versions
            pos = mne.viz.plot_topomap._prepare_topomap_plot(info, picks=picks)[0]
            logger.info(f"Got clean positions from _prepare_topomap_plot: {pos.shape}")
            return pos, picks
        except AttributeError:
            # For older MNE versions, try alternative method
            try:
                pos = mne.viz.plot_topomap._prepare_topomap_plot(info, picks=picks)[0]
                logger.info(f"Got clean positions from alternative method: {pos.shape}")
                return pos, picks
            except:
                pass
        
        # Fallback to manual position extraction
        try:
            layout = mne.channels.find_layout(info, ch_type='eeg')
            if layout and hasattr(layout, 'pos') and layout.pos is not None:
                pos = layout.pos
                logger.info(f"Got positions from layout: {pos.shape}")
                # Handle 4D positions by taking first 2 dimensions
                if pos.shape[1] == 4:
                    pos = pos[:, :2]
                    logger.info("Converted 4D positions to 2D")
                elif pos.shape[1] == 3:
                    pos = pos[:, :2]
                    logger.info("Converted 3D positions to 2D")
                return pos, picks
        except Exception as e2:
            logger.error(f"Layout fallback also failed: {e2}")
        
        # Final fallback: create custom 10-20 positions
        logger.warning("Using custom 10-20 positions as final fallback")
        return create_custom_1020_positions(info.ch_names), picks
        
    except Exception as e:
        logger.error(f"All position methods failed: {e}")
        return create_custom_1020_positions(info.ch_names), picks

def clean_channel_name(name):
    """Clean channel names to clinical 10-20 format"""
    name = str(name).strip()
    
    # Remove suffixes first
    for suffix in ['-LE', '-RE', '-REF', '-M1', '-M2', '-A1', '-A2']:
        name = name.replace(suffix, '')
    
    # Convert to uppercase for processing
    name_upper = name.upper()
    
    # Convert old to new nomenclature
    replacements = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    for old, new in replacements.items():
        name_upper = name_upper.replace(old, new)
    
    # Convert to proper MNE case format
    mne_case_map = {
        'FP1': 'Fp1', 'FP2': 'Fp2',
        'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
        'F3': 'F3', 'F4': 'F4', 'F7': 'F7', 'F8': 'F8',
        'C3': 'C3', 'C4': 'C4', 'T7': 'T7', 'T8': 'T8',
        'P3': 'P3', 'P4': 'P4', 'P7': 'P7', 'P8': 'P8',
        'O1': 'O1', 'O2': 'O2'
    }
    
    return mne_case_map.get(name_upper, name_upper.title())

def get_clinical_positions(channel_names):
    """Get clinical-grade electrode positions"""
    positions = []
    valid_channels = []
    
    for ch in channel_names:
        clean_ch = clean_channel_name(ch)
        if clean_ch in CLINICAL_1020_POSITIONS:
            positions.append(CLINICAL_1020_POSITIONS[clean_ch])
            valid_channels.append(clean_ch)
        else:
            logger.warning(f"Channel {ch} ({clean_ch}) not found in clinical 10-20 system")
    
    return np.array(positions), valid_channels

def clinical_interpolation(values, positions, resolution=128):
    """
    High-quality clinical interpolation matching NeuroGuide standards
    """
    # Create high-resolution grid that matches plot limits exactly
    xi = np.linspace(-1.3, 1.3, resolution)
    yi = np.linspace(-1.3, 1.3, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Create circular mask (clinical head shape)
    mask = np.sqrt(Xi**2 + Yi**2) <= 1.0
    
    # Interpolate using cubic method (clinical standard)
    Zi = griddata(positions, values, (Xi, Yi), method='cubic', fill_value=0)
    
    # Apply circular mask
    Zi[~mask] = 0
    
    return Xi, Yi, Zi

def plot_clean_topomap(data, info, title='', cmap=DEFAULT_CMAP, vlim=DEFAULT_VLIM, 
                       show_sensors=True, contours=True, head_outline=True, is_zscore=False, 
                       paradox_theme=True, label_mode='auto'):
    """
    Plot a clinical-grade topomap matching NeuroGuide/ClinicalQ standards
    
    Args:
        data: 1D array of values (must match EEG channels in info)
        info: MNE Info object with channel locations
        title: Plot title
        cmap: Colormap name
        vlim: (vmin, vmax) for color scaling (None for auto)
        show_sensors: Whether to show electrode markers
        contours: Whether to show contour lines
        head_outline: Whether to show head outline
        is_zscore: Whether values are z-scores (affects color scaling)
    """
    # Validate input
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if data.ndim != 1:
        raise ValueError(f"Data must be 1D, got shape {data.shape}")
    
    # Get clinical positions using channel names
    channel_names = info.ch_names
    positions, valid_channels = get_clinical_positions(channel_names)
    
    if len(positions) == 0:
        raise ValueError("No valid clinical channels found")
    
    # Ensure data matches valid channels
    if len(data) != len(valid_channels):
        logger.warning(f"Adjusting data from {len(data)} to {len(valid_channels)} channels")
        data = data[:len(valid_channels)]
    
    data = np.array(data, dtype=float)
    
    # Set clinical color limits
    if vlim is None or vlim == DEFAULT_VLIM:
        if is_zscore:
            vmin, vmax = -3.0, 3.0
            cmap = 'RdBu_r'
        else:
            vmin, vmax = np.percentile(data, [5, 95])  # Clinical percentiles
    else:
        vmin, vmax = vlim
    
    # Create figure with clinical styling
    figsize = (10, 8)
    # bg + cmap
    if paradox_theme:
        fig, ax = plt.subplots(figsize=figsize, facecolor=CYBER_BG)
        ax.set_facecolor(CYBER_BG)
        if is_zscore:
            cmap = PARADOX_CMAP_DIV
        elif cmap in (None, 'RdBu_r', DEFAULT_CMAP):
            cmap = PARADOX_CMAP_SEQ
    else:
        fig, ax = plt.subplots(figsize=figsize, facecolor='black')
        ax.set_facecolor('black')
    
    # Perform clinical interpolation
    Xi, Yi, Zi = clinical_interpolation(data, positions, resolution=128)
    
    # Create clinical-style contour plot
    # Ensure vmin and vmax are different and valid
    if vmax <= vmin:
        vmax = vmin + 1e-6
    
    levels = np.linspace(vmin, vmax, 20)
    # Ensure levels are strictly increasing
    levels = np.unique(levels)
    if len(levels) < 2:
        levels = np.array([vmin, vmax])
    
    contour_filled = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, 
                                vmin=vmin, vmax=vmax, extend='both')
    
    # Add clinical contour lines
    if contours:
        line_col = '#000' if not paradox_theme else '#1a2a44'
        contour_lines = ax.contour(Xi, Yi, Zi, levels=10, colors=line_col, 
                                  linewidths=0.6, alpha=0.7, zorder=4)
    
    # Draw clinical head outline
    if head_outline:
        if paradox_theme:
            _neon_glow_circle(ax, (0,0), 1.0, NEON_CYAN)
            # nose
            ax.plot([0,-0.1,0.1,0], [1.0,1.15,1.15,1.0],
                    color=NEON_CYAN, linewidth=1.8, alpha=0.8, zorder=3,
                    path_effects=_halo(lw=4.0, color=CYBER_BG, alpha=0.9))
            # ears
            for sx in (-1.0, 1.0):
                e = plt.Circle((sx,0), 0.10, fill=False, color=NEON_CYAN, linewidth=1.4, alpha=0.8, zorder=3)
                ax.add_patch(e)
        else:
            head_circle = plt.Circle((0, 0), 1.0, fill=False, color='white', linewidth=3)
            ax.add_patch(head_circle)
            ax.plot([0,-0.1,0.1,0], [1.0,1.15,1.15,1.0], 'white', linewidth=2)
            ax.add_patch(plt.Circle((-1.0,0),0.1,fill=False,color='white',linewidth=2))
            ax.add_patch(plt.Circle((1.0,0),0.1,fill=False,color='white',linewidth=2))
    
    # Plot electrode positions
    if show_sensors:
        dot_edge = CYBER_BG if paradox_theme else 'black'
        ax.scatter(positions[:,0], positions[:,1], c=CYBER_FG, s=28 if not paradox_theme else 36,
                   edgecolors=dot_edge, linewidths=0.8, zorder=10, alpha=0.85)

        # Label coloring
        for i, (pos, ch) in enumerate(zip(positions, valid_channels)):
            if is_zscore:
                color = _severity_color(data[i])
            else:
                color = _region_color(ch)
            ax.text(pos[0], pos[1] - 0.08, ch,
                    ha='center', va='top',
                    color=color, fontsize=9 if not paradox_theme else 10, fontweight='bold',
                    zorder=11,
                    path_effects=_halo(lw=3.0, color=CYBER_BG if paradox_theme else 'black', alpha=1.0))
    
    # Clinical colorbar
    cbar = plt.colorbar(contour_filled, ax=ax, shrink=0.84, aspect=28, pad=0.02)
    txt_color = CYBER_FG if paradox_theme else 'white'
    cbar.ax.tick_params(colors=txt_color, labelsize=12)
    cbar.outline.set_edgecolor('#2a3c5c' if paradox_theme else 'white')
    cbar.outline.set_linewidth(1.2 if paradox_theme else 1.0)
    cbar.set_label('Z-Score' if is_zscore else 'Power (µV²)',
                   color=txt_color, fontsize=14, fontweight='bold',
                   labelpad=8)
    
    # Clinical styling
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Clear any unwanted elements that might cause visual artifacts
    for artist in ax.get_children():
        if hasattr(artist, 'get_position') and hasattr(artist, 'get_text'):
            pos = artist.get_position()
            if pos[0] > 1.3 or pos[0] < -1.3 or pos[1] > 1.3 or pos[1] < -1.3:
                artist.remove()
    
    ax.set_title(title, color=txt_color, fontsize=18, fontweight='bold', pad=18,
                 path_effects=_halo(lw=6.0, color=CYBER_BG, alpha=1.0) if paradox_theme else None)
    
    plt.tight_layout()
    logger.info(f"Created clinical topomap with {len(valid_channels)} channels")
    return fig

def create_clinicalQ_grid(site_metrics, title="ClinicalQ Analysis"):
    """
    Create a professional, readable ClinicalQ analysis report
    
    Args:
        site_metrics: list of dicts like:
          [{'site':'Fp1','metric':'Alpha EC','z':1.2,'value':0.85},
           {'site':'Cz','metric':'Theta/Beta','z':2.7,'value':2.9}, ...]
        title: Grid title
           
    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create larger figure for better readability
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(CYBER_BG)
    ax.set_facecolor(CYBER_BG)
    
    # Organize data by metric and site
    sites = sorted({d['site'] for d in site_metrics})
    metrics = sorted({d['metric'] for d in site_metrics})
    
    # Create professional table layout
    row_height = 1.2
    col_width = 2.4
    start_y = len(metrics) * row_height
    
    # Title
    ax.text(len(sites) * col_width / 2, start_y + 0.8, title, 
            ha='center', va='center', color=CYBER_FG, fontsize=20, fontweight='bold',
            path_effects=_halo(lw=6.0, color=CYBER_BG, alpha=1.0))
    
    # Headers - Sites
    for i, site in enumerate(sites):
        x = i * col_width + col_width/2
        y = start_y + 0.2
        
        # Site header background
        rect = patches.Rectangle((i * col_width + 0.1, start_y - 0.1), 
                                col_width - 0.2, 0.6, 
                                facecolor='#1a2a44', edgecolor=NEON_CYAN, 
                                linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        
        ax.text(x, y, site, ha='center', va='center', 
                color=NEON_CYAN, fontsize=14, fontweight='bold',
                path_effects=_halo(lw=3.0, color=CYBER_BG, alpha=1.0))
    
    # Data rows
    for row, metric in enumerate(metrics):
        y = start_y - (row + 1) * row_height
        
        # Metric label
        ax.text(-0.3, y, metric, ha='right', va='center', 
                color=CYBER_FG, fontsize=12, fontweight='bold',
                path_effects=_halo(lw=2.5, color=CYBER_BG, alpha=1.0))
        
        # Data cells
        for col, site in enumerate(sites):
            x = col * col_width + col_width/2
            
            # Find data for this metric/site combination
            site_data = next((d for d in site_metrics 
                            if d['metric'] == metric and d['site'] == site), None)
            
            if site_data:
                z_score = site_data.get('z', 0.0)
                value = site_data.get('value', 0.0)
                
                # Determine colors and badge
                if abs(z_score) >= 2.58:
                    bg_color = NEON_RED
                    badge = 'SEVERE'
                    text_color = 'white'
                elif abs(z_score) >= 2.0:
                    bg_color = NEON_ORANGE
                    badge = 'ABNORMAL'
                    text_color = 'black'
                elif abs(z_score) >= 1.5:
                    bg_color = NEON_YELLOW
                    badge = 'BORDERLINE'
                    text_color = 'black'
                else:
                    bg_color = '#90EE90'
                    badge = 'NORMAL'
                    text_color = 'black'
                
                # Cell background with neon glow
                rect = patches.Rectangle((col * col_width + 0.1, y - row_height/2 + 0.1), 
                                        col_width - 0.2, row_height - 0.2, 
                                        facecolor=bg_color, alpha=0.3, 
                                        edgecolor=bg_color, linewidth=2)
                ax.add_patch(rect)
                
                # Value text (larger, more readable)
                ax.text(x, y + 0.15, f"{value:.2f}", ha='center', va='center', 
                        color=CYBER_FG, fontsize=16, fontweight='bold',
                        path_effects=_halo(lw=3.0, color=CYBER_BG, alpha=1.0))
                
                # Z-score text
                ax.text(x, y - 0.05, f"z = {z_score:.2f}", ha='center', va='center', 
                        color=CYBER_FG, fontsize=12, fontweight='normal',
                        path_effects=_halo(lw=2.5, color=CYBER_BG, alpha=1.0))
                
                # Badge
                ax.text(x, y - 0.25, badge, ha='center', va='center', 
                        color=bg_color, fontsize=10, fontweight='bold',
                        path_effects=_halo(lw=3.0, color=CYBER_BG, alpha=1.0))
                
        else:
                # Empty cell
                rect = patches.Rectangle((col * col_width + 0.1, y - row_height/2 + 0.1), 
                                        col_width - 0.2, row_height - 0.2, 
                                        facecolor='gray', alpha=0.2, 
                                        edgecolor='gray', linewidth=1)
                ax.add_patch(rect)
                ax.text(x, y, 'N/A', ha='center', va='center', 
                        color='gray', fontsize=12, style='italic')
    
    # Legend (moved to bottom for better layout)
    legend_y = -0.8
    legend_items = [
        ('NORMAL', '#90EE90', '<1.5σ'),
        ('BORDERLINE', NEON_YELLOW, '≥1.5σ'),
        ('ABNORMAL', NEON_ORANGE, '≥2.0σ'),
        ('SEVERE', NEON_RED, '≥2.58σ')
    ]
    
    ax.text(len(sites) * col_width / 2, legend_y - 0.3, 'Clinical Significance Levels', 
            ha='center', va='center', color=CYBER_FG, fontsize=14, fontweight='bold',
            path_effects=_halo(lw=4.0, color=CYBER_BG, alpha=1.0))
    
    for i, (label, color, threshold) in enumerate(legend_items):
        x = i * 3.5 + 1.5
        
        # Legend box
        rect = patches.Rectangle((x - 0.4, legend_y - 0.15), 0.8, 0.3, 
                                facecolor=color, alpha=0.4, 
                                edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        ax.text(x, legend_y, f"{label}\n{threshold}", ha='center', va='center', 
                color=color, fontsize=10, fontweight='bold',
                path_effects=_halo(lw=2.5, color=CYBER_BG, alpha=1.0))
    
    # Set limits and remove axes
    ax.set_xlim(-1, len(sites) * col_width)
    ax.set_ylim(legend_y - 0.6, start_y + 1.2)
    ax.axis('off')
    
    plt.tight_layout()
    logger.info(f"Created professional ClinicalQ report with {len(metrics)} metrics and {len(sites)} sites")
    return fig

# ========== SPECIALIZED FUNCTIONS ==========

def create_zscore_topomap(z_scores, channel_names, title, clinical_thresholds=True):
    """Create Z-score topomap with clinical significance indicators"""
    try:
        # Clean channel names
        clean_names = [clean_channel_name(name) for name in channel_names]
        
        # Create info object
        info = mne.create_info(
            ch_names=clean_names,
            sfreq=250,
            ch_types=['eeg'] * len(clean_names)
        )
        
        # Create figure with clinical standards
        fig = plot_clean_topomap(
            z_scores, info, title,
            cmap='RdBu_r',
            vlim=None,  # Auto-set to (-3, 3) for z-scores
            show_sensors=True,
            contours=True,
            head_outline=True,
            is_zscore=True,  # Enables clinical z-score scaling
            paradox_theme=True
        )
        
        if clinical_thresholds and fig is not None:
            # Clinical significance indicators removed to prevent unwanted dots
            # The topomap colors already indicate clinical significance through the color scale
            pass
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Z-score topomap: {e}")
        return None

def create_clinical_topomap_grid(values_dict, channel_names, condition='Unknown', 
                                clinical_analysis=True, z_scores_dict=None):
    """Create a grid of clinical topographical maps"""
    try:
        logger.info(f"Creating clinical grid with {len(values_dict)} metrics")
        
        n_metrics = len(values_dict)
        if n_metrics == 0:
            logger.warning("No metrics provided")
            return None
        
        # Determine grid layout
        n_cols = min(3, n_metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), 
                                facecolor='#000000')
        fig.patch.set_facecolor('#000000')
        
        # Handle different axes configurations
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Create info object once
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=250,
            ch_types=['eeg'] * len(channel_names)
        )
        
        # Plot each metric
        for idx, (metric_name, values) in enumerate(values_dict.items()):
            logger.info(f"Processing metric {idx+1}/{n_metrics}: {metric_name}")
            
            row = idx // n_cols
            col = idx % n_cols
            
            # Get the correct subplot
            if n_metrics == 1:
                ax = axes[0] if isinstance(axes, list) else axes
            elif n_rows == 1:
                ax = axes[col] if n_cols > 1 else axes
            else:
                ax = axes[row, col]
            
            try:
                # Ensure values is a numpy array and has correct length
                values = np.array(values, dtype=float)
                logger.info(f"Metric {metric_name}: values shape={values.shape}, channels={len(channel_names)}")
                
                if len(values) != len(channel_names):
                    logger.warning(f"Values length {len(values)} doesn't match channels {len(channel_names)}, adjusting")
                    # Pad or truncate to match channel count
                    if len(values) < len(channel_names):
                        values = np.pad(values, (0, len(channel_names) - len(values)), mode='constant', constant_values=0)
                    else:
                        values = values[:len(channel_names)]
                    logger.info(f"Adjusted values shape: {values.shape}")
                
                # Get channel positions directly for this subplot
                positions, valid_channels = get_clinical_positions(channel_names)
                
                if len(positions) == 0:
                    logger.warning(f"No valid positions found for {metric_name}")
                    ax.text(0.5, 0.5, f"No Data\n{metric_name}", 
                           ha='center', va='center', transform=ax.transAxes,
                           color='#ffff00', fontsize=12)
                    ax.axis('off')
                    continue
                
                # Adjust values to match valid positions
                if len(values) > len(valid_channels):
                    values = values[:len(valid_channels)]
                elif len(values) < len(valid_channels):
                    # Pad with zeros if needed
                    values = np.pad(values, (0, len(valid_channels) - len(values)), mode='constant', constant_values=0)
                
                try:
                    # Perform interpolation directly on this subplot
                    Xi, Yi, Zi = clinical_interpolation(values, positions, resolution=64)
                    
                    # Plot directly on the subplot
                    vmin, vmax = np.percentile(values, [5, 95])
                    if vmax <= vmin:
                        vmax = vmin + 1e-6
                    
                    # Create contour plot
                    levels = np.linspace(vmin, vmax, 15)
                    levels = np.unique(levels)
                    if len(levels) < 2:
                        levels = np.array([vmin, vmax])
                    
                    contour_filled = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='viridis', 
                                               vmin=vmin, vmax=vmax, extend='both')
                    
                    # Add head outline
                    head_circle = plt.Circle((0, 0), 1.0, fill=False, color='white', linewidth=2)
                    ax.add_patch(head_circle)
                    
                    # Add electrode positions
                    ax.scatter(positions[:,0], positions[:,1], c='white', s=20, 
                             edgecolors='black', linewidths=0.5, zorder=10)
                    
                    # Set limits and styling
                    ax.set_xlim(-1.2, 1.2)
                    ax.set_ylim(-1.2, 1.2)
                    ax.set_aspect('equal')
                    
                    # Add title
                    ax.set_title(f"{metric_name}\n({condition})", 
                               color='#ffffff', fontsize=11, weight='bold')
                    
                except Exception as interp_error:
                    logger.warning(f"Interpolation error for {metric_name}: {interp_error}")
                    # Fallback: simple scatter plot
                    ax.scatter(positions[:,0], positions[:,1], c=values[:len(positions)], 
                             cmap='viridis', s=100, edgecolors='white', linewidths=1)
                    ax.set_title(f"{metric_name} (Scatter)\n({condition})", 
                               color='#ffffff', fontsize=11, weight='bold')
                    ax.set_xlim(-1.2, 1.2)
                    ax.set_ylim(-1.2, 1.2)
                    ax.set_aspect('equal')
                
            except Exception as e:
                logger.warning(f"Error creating topomap for {metric_name}: {e}")
                ax.text(0.5, 0.5, f"Error\n{metric_name}", 
                       ha='center', va='center', transform=ax.transAxes,
                       color='#ff0000', fontsize=12)
            
            # Clean up subplot
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            
            if n_metrics == 1:
                continue
            elif n_rows == 1:
                axes[0, col].axis('off')
            else:
                axes[row, col].axis('off')
        
        # Main title
        fig.suptitle(f"Clinical Topographical Analysis - {condition}", 
                    color='#ffffff', fontsize=16, y=0.98, weight='bold')
        
        # Finalize layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        logger.info(f"Successfully created clinical grid with {n_metrics} metrics")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating clinical grid: {e}")
        return None

def create_professional_topomap(values, channel_names, title, cmap='viridis', 
                               show_sensors=True, show_contours=True, 
                               clinical_indicators=False, clinical_thresholds=None):
    """Create a professional QEEG-style topographical brain map"""
    try:
        # Clean channel names
        clean_names = [clean_channel_name(name) for name in channel_names]
        
        # Create info object
        info = mne.create_info(
            ch_names=clean_names,
            sfreq=250,
            ch_types=['eeg'] * len(clean_names)
        )
        
        # Determine if this is z-score data and set appropriate scaling
        is_zscore = 'z-score' in title.lower() or 'zscore' in title.lower()
        
        # Color limits will be auto-set by plot_clean_topomap based on is_zscore
        vlim = None
        
        # Create figure with clinical standards
        fig = plot_clean_topomap(
            values, info, title,
            cmap=cmap,
            vlim=vlim,
            show_sensors=show_sensors,
            contours=show_contours,
            head_outline=True,
            is_zscore=is_zscore,  # Enables appropriate scaling
            paradox_theme=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating professional topomap: {e}")
        return None

# ========== UTILITY FUNCTIONS ==========

def debug_info_shape(data, info):
    """Debug utility to check data and info compatibility"""
    try:
        picks = mne.pick_types(info, eeg=True)
        print("EEG Channels:", picks)
        print("Data shape:", data.shape)
        print("Info channels:", len(info['ch_names']))
        print("Channel names:", info['ch_names'])
        if info.get_montage():
            print("Montage attached: YES")
            print("Montage type:", info.get_montage().kind)
        else:
            print("Montage attached: NO")
    except Exception as e:
        print(f"Debug error: {e}")

def save_topomap(fig, filepath, dpi=300, facecolor='black'):
    """Save a topographical map figure with proper styling"""
    try:
        if fig is not None:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                       facecolor=facecolor, edgecolor='none')
            logger.debug(f"Topographical map saved to: {filepath}")
            return True
        else:
            logger.warning(f"Cannot save None figure to: {filepath}")
            return False
    except Exception as e:
        logger.error(f"Error saving topographical map to {filepath}: {e}")
        return False

# ========== TEST CALL (if standalone run) ==========

if __name__ == "__main__":
    # Dummy test with 19-channel dummy z-scores
    dummy_info = mne.create_info(ch_names=[
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'O2'
    ], sfreq=256, ch_types='eeg')

    dummy_data = np.random.normal(0, 1, 19)
    ensure_montage(dummy_info)
    
    print("Testing clean topomap creation...")
    fig = plot_clean_topomap(dummy_data, dummy_info, title='Test Clean Topomap')
    
    if fig:
        print("✅ Test successful! Saving test image...")
        fig.savefig('test_clean_topomap.png', dpi=150, bbox_inches='tight', facecolor='black')
        print("📁 Saved as: test_clean_topomap.png")
        plt.close(fig)
    else:
        print("❌ Test failed!")
