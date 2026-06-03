import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

sys.path.append('.')
import diagnostics_lib as dl

def load_tif(path):
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    return img

def main():
    base_dir = r"c:\Users\ILILUser\Desktop\Stability_Analysis\20260416\SideImaging"
    
    # Load background (shot 1) and signal (shot 30)
    bg_path = os.path.join(base_dir, "SideImaging_20260416_1.tif")
    sig_path = os.path.join(base_dir, "SideImaging_20260416_60.tif")
    
    if not os.path.exists(bg_path) or not os.path.exists(sig_path):
        print("Images not found!")
        return
        
    bg_img = load_tif(bg_path)
    sig_img = load_tif(sig_path)
    
    print("Images loaded. bg_img shape:", bg_img.shape, "sig_img shape:", sig_img.shape)
    
    # Sottrazione del background
    res_bg = dl.subtract_background(sig_img, bg_img)
    cleaned_img = res_bg['cleaned_image']
    
    # Analisi Side View
    THRESHOLD_FRAC = 0.1
    print("Running analyze_plasma_channel_side...")
    res = dl.analyze_plasma_channel_side(
        cleaned_img, raw_image=sig_img,
        threshold_fraction=THRESHOLD_FRAC,
    )
    
    debug = res.get('debug_data', {})
    
    z_profile = debug.get('z_profile_robust')
    z_c = res.get('Plasma_Z_Position')
    y_c = res.get('Plasma_Y_Position')
    max_intensity = res.get('Max_Intensity', 0)
    threshold_val = max_intensity * THRESHOLD_FRAC
    plasma_length = res.get('Plasma_Length')
    plasma_start = debug.get('plasma_start')
    plasma_end = debug.get('plasma_end')
    nozzle_contour = debug.get('nozzle_contour')
    nozzle_tip = debug.get('nozzle_tip')
    
    # Axis extraction data (new format: tuple)
    axis_coords = debug.get('axis_coords')
    if axis_coords is not None:
        axis_z, axis_y = axis_coords
    else:
        axis_z, axis_y = None, None
    
    zc_str = f"{z_c:.1f}" if z_c is not None and not np.isnan(z_c) else "None"
    yc_str = f"{y_c:.1f}" if y_c is not None and not np.isnan(y_c) else "None"
    print(f"Results: Z_c={zc_str}, Y_c={yc_str}, Plasma Length={plasma_length} px")
    print(f"  Nozzle Distance Y: {res.get('Nozzle_Distance_Y_mm', 'N/A')} mm")
    print(f"  Side Beam Angle:   {res.get('Side_Beam_Angle_mrad', 'N/A')} mrad")
    print(f"  Side Channel Width:{res.get('Side_Channel_Width_mean', 'N/A')} px")
    
    # Output directory
    out_dir = r"c:\Users\ILILUser\Desktop\Stability_Analysis\debug_output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Grafico 1: Side View Analysis - Immagine con nozzle e asse
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(cleaned_img, cmap='jet', origin='upper', vmin=0, vmax=np.percentile(cleaned_img, 99))
    
    if nozzle_contour is not None:
        ax.plot(nozzle_contour[:, 1], nozzle_contour[:, 0], color='cyan', linewidth=1.5, alpha=0.8, label='Nozzle contour')
    if nozzle_tip is not None:
        ax.plot(nozzle_tip[0], nozzle_tip[1], 'rx', markersize=12, markeredgewidth=2, label='Nozzle tip')
    if axis_z is not None and axis_y is not None and len(axis_z) > 0:
        ax.plot(axis_z, axis_y, 'w.', markersize=2, alpha=0.6, label='Fitted Channel Axis')
    if z_c is not None and y_c is not None and not np.isnan(z_c) and not np.isnan(y_c):
        ax.plot(z_c, y_c, 'y+', markersize=14, markeredgewidth=2.5, label='Centroid (Z_c, Y_c)')
        
    # Mostra l'area considerata come lunghezza plasma
    if plasma_start is not None and plasma_end is not None:
        if not np.isnan(plasma_start) and not np.isnan(plasma_end):
            ax.axvspan(plasma_start, plasma_end, color='white', alpha=0.2, label='Plasma Length region')

    ax.set_title('Side Imaging - Channel Analysis (Axis Extraction)')
    ax.legend(loc='lower right')
    fig.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_side_view_analysis.png"), dpi=150)
    plt.close(fig)
    print("Saved 05_side_view_analysis.png")
    
    # Grafico 2: Z Profile e Threshold
    if z_profile is not None:
        fig, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(z_profile, label='Robust Z profile (99.5th pctl)')
        if threshold_val is not None:
            ax2.axhline(threshold_val, color='r', ls='--', alpha=0.7, label=f'Threshold ({threshold_val:.1f})')
        if z_c is not None and not np.isnan(z_c):
            ax2.axvline(z_c, color='gold', ls=':', alpha=0.7, label=f'Z_c ({z_c:.1f})')
        if plasma_start is not None and plasma_end is not None:
            if not np.isnan(plasma_start) and not np.isnan(plasma_end):
                ax2.axvspan(plasma_start, plasma_end, color='green', alpha=0.2, label='Plasma Length span')
            
        ax2.set_title('Robust Z Profile for Length Calculation')
        ax2.set_xlabel('Z (pixels)')
        ax2.set_ylabel('Intensity')
        ax2.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "06_side_view_profile.png"), dpi=150)
        plt.close(fig)
        print("Saved 06_side_view_profile.png")
    
    print(f"Results saved in {out_dir}")

if __name__ == '__main__':
    main()
