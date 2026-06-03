import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from scipy import ndimage
import os

sys.path.append('.')
import diagnostics_lib as dl

def load_tif(path):
    return np.array(Image.open(path))

def main():
    base_dir = r"c:\Users\ILILUser\Desktop\Stability_Analysis\20260415\TopView"
    
    # Load background (shot 1) and signal (shot 30)
    bg_path = os.path.join(base_dir, "TopView_20260415_1.tif")
    sig_path = os.path.join(base_dir, "TopView_20260415_30.tif")
    
    if not os.path.exists(bg_path) or not os.path.exists(sig_path):
        print("Images not found!")
        return
        
    bg_img = load_tif(bg_path)
    sig_img = load_tif(sig_path)
    
    print("Images loaded. bg_img shape:", bg_img.shape, "sig_img shape:", sig_img.shape)
    
    # Sottrazione del background usando l'immagine _1.tif
    res_bg = dl.subtract_background(sig_img, bg_img)
    cleaned_img = res_bg['cleaned_image']
    
    # Analyze top view
    print("Running analyze_plasma_channel_top...")
    res = dl.analyze_plasma_channel_top(
        cleaned_img,
        threshold_fraction=0.1,
        fwhm_fraction_saturated=0.8
    )
    
    debug = res['debug_data']
    axis_z, axis_y = debug['axis_coords']
    axis_y_err = debug['axis_coords_err']
    axis_width = debug['axis_width']
    axis_y_left = debug['axis_y_left']
    axis_y_right = debug['axis_y_right']
    
    # Print summary
    print(f"  Beam Angle:   {res['Top_Beam_Angle_mrad']:.3f} ± "
          f"{res['Top_Beam_Angle_mrad_err']:.3f} mrad")
    print(f"  Channel Width (mean FWHM): {res['Top_Channel_Width_mean']:.1f} px")
    print(f"  Total Plasma Length: {res['Plasma_Length']:.1f} px")
    print(f"  Saturated Core Length: {res['Top_Saturated_Length']:.1f} px")
    
    # Output directory
    out_dir = r"c:\Users\ILILUser\Desktop\Stability_Analysis\debug_output"
    os.makedirs(out_dir, exist_ok=True)
    
    # ── 1. Image with axis and FWHM band ──
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(cleaned_img, cmap='jet', origin='upper', vmin=0, vmax=150)
    
    if len(axis_z) > 0:
        valid = np.isfinite(axis_y_left) & np.isfinite(axis_y_right)
        
        # Identifica le colonne saturate vs non saturate per un debug più esplicativo
        sat_thresh = 254.0
        is_sat = np.array([np.any(cleaned_img[:, int(z)] >= sat_thresh) for z in axis_z])
        
        if np.any(valid):
            # Rimuoviamo la "striscia bianca" (fill_between) e disegniamo i bordi esatti
            ax.plot(axis_z[valid], axis_y_left[valid],
                    color='magenta', linestyle='--', linewidth=1.5, label='FWHM Left edge')
            ax.plot(axis_z[valid], axis_y_right[valid],
                    color='cyan', linestyle='--', linewidth=1.5, label='FWHM Right edge')
            
        # Centre line differenziata tra punti saturi e non saturi
        ax.plot(axis_z, axis_y, color='white', linewidth=1.0, alpha=0.5, label='Centre axis (fit)')
        
        # Disegna i punti centrali con colori diversi a seconda della saturazione
        ax.scatter(axis_z[~is_sat], axis_y[~is_sat], color='white', s=2, label='Unsaturated centre')
        if np.any(is_sat):
            ax.scatter(axis_z[is_sat], axis_y[is_sat], color='red', s=5, label='Saturated centre')
        
        # Linear fit line (beam angle)
        if np.isfinite(res['Top_Beam_Angle_mrad']):
            coeffs = np.polyfit(axis_z, axis_y, deg=1)
            fit_line = np.polyval(coeffs, axis_z)
            ax.plot(axis_z, fit_line, color='yellow', linestyle=':', linewidth=2.0,
                    label=f'Linear fit ({res["Top_Beam_Angle_mrad"]:.2f} mrad)')
    
    ax.set_title(f'Top View - Shot 30  |  Angle: '
                 f'{res["Top_Beam_Angle_mrad"]:.2f} mrad  |  '
                 f'FWHM: {res["Top_Channel_Width_mean"]:.1f} px\n'
                 f'Total Length: {res["Plasma_Length"]:.0f} px  |  Sat. Length: {res["Top_Saturated_Length"]:.0f} px')
    
    # Sposta la legenda fuori se necessario o lasciala in alto a destra ma più piccola
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_top_view_axis.png"), dpi=150)
    plt.close(fig)
    print("Saved 01_top_view_axis.png")
    
    # ── 2. Profile plots: one saturated, one unsaturated ──
    # Find examples
    sat_thresh = 254.0
    z_sat = None
    z_unsat = None
    for i, z in enumerate(axis_z):
        col = cleaned_img[:, int(z)]
        if z_sat is None and np.any(col >= sat_thresh):
            z_sat = int(z)
            idx_sat = i
        if z_unsat is None and not np.any(col >= sat_thresh):
            z_unsat = int(z)
            idx_unsat = i
        if z_sat is not None and z_unsat is not None:
            break
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ── Left panel: Saturated column ──
    if z_sat is not None:
        profile = cleaned_img[:, z_sat]
        y_px = np.arange(len(profile))
        axes[0].plot(y_px, profile, 'k.-', markersize=2, label='Data')
        
        # Half-max level
        half_max = 0.6 * 255.0
        axes[0].axhline(half_max, color='orange', linestyle=':', linewidth=1,
                        label=f'60% × 255 = {half_max:.0f}')
        
        # FWHM edges
        yl = axis_y_left[idx_sat]
        yr = axis_y_right[idx_sat]
        if np.isfinite(yl):
            axes[0].axvline(yl, color='blue', linestyle='--', linewidth=1,
                            label=f'Left edge: {yl:.1f}')
        if np.isfinite(yr):
            axes[0].axvline(yr, color='blue', linestyle='--', linewidth=1,
                            label=f'Right edge: {yr:.1f}')
        
        # Centre
        axes[0].axvline(axis_y[idx_sat], color='red', linewidth=2,
                        label=f'Centre: {axis_y[idx_sat]:.1f}')
        
        axes[0].set_title(f'Saturated Column (Z={z_sat})\n'
                          f'Width = {axis_width[idx_sat]:.1f} px')
        axes[0].set_xlabel('Y (px)')
        axes[0].set_ylabel('Intensity')
        axes[0].legend(fontsize=7)
        
        # Zoom around the peak
        center = int(axis_y[idx_sat])
        margin = 80
        axes[0].set_xlim(center - margin, center + margin)
    else:
        axes[0].set_title('No saturated column found')
    
    # ── Right panel: Unsaturated column ──
    if z_unsat is not None:
        profile = cleaned_img[:, z_unsat]
        col_smooth = ndimage.gaussian_filter1d(profile, sigma=1.0)
        y_px = np.arange(len(profile))
        
        axes[1].plot(y_px, profile, 'k.-', markersize=2, label='Data')
        axes[1].plot(y_px, col_smooth, 'gray', linewidth=1, alpha=0.7,
                     label='Smoothed')
        
        # Half-max level
        peak_val = col_smooth[int(np.argmax(col_smooth))]
        half_max = 0.5 * peak_val
        axes[1].axhline(half_max, color='orange', linestyle=':', linewidth=1,
                        label=f'50% × {peak_val:.0f} = {half_max:.0f}')
        
        # FWHM edges
        yl = axis_y_left[idx_unsat]
        yr = axis_y_right[idx_unsat]
        if np.isfinite(yl):
            axes[1].axvline(yl, color='blue', linestyle='--', linewidth=1,
                            label=f'Left edge: {yl:.1f}')
        if np.isfinite(yr):
            axes[1].axvline(yr, color='blue', linestyle='--', linewidth=1,
                            label=f'Right edge: {yr:.1f}')
        
        # Centre
        axes[1].axvline(axis_y[idx_unsat], color='red', linewidth=2,
                        label=f'Centre: {axis_y[idx_unsat]:.1f}')
        
        axes[1].set_title(f'Unsaturated Column (Z={z_unsat})\n'
                          f'Width = {axis_width[idx_unsat]:.1f} px')
        axes[1].set_xlabel('Y (px)')
        axes[1].set_ylabel('Intensity')
        axes[1].legend(fontsize=7)
        
        # Zoom around the peak
        center = int(axis_y[idx_unsat])
        margin = 80
        axes[1].set_xlim(center - margin, center + margin)
    else:
        axes[1].set_title('No unsaturated column found')
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_top_view_profiles.png"), dpi=150)
    plt.close(fig)
    print("Saved 02_top_view_profiles.png")
    print(f"Results saved in {out_dir}")

if __name__ == '__main__':
    main()
