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
    base_dir = r"c:\Users\ILILUser\Desktop\Stability_Analysis\20260415\Pointing_Lanex"
    
    # Load background (shot 1) and signal (shot 30)
    bg_path = os.path.join(base_dir, "Pointing_Lanex_20260415_1.tiff")
    sig_path = os.path.join(base_dir, "Pointing_Lanex_20260415_30.tiff")
    
    if not os.path.exists(bg_path) or not os.path.exists(sig_path):
        print("Images not found!")
        return
        
    bg_img = load_tif(bg_path)
    sig_img = load_tif(sig_path)
    
    print("Images loaded. bg_img shape:", bg_img.shape, "sig_img shape:", sig_img.shape)
    
    # Sottrazione del background
    res_bg = dl.subtract_background(sig_img, bg_img)
    cleaned_img = res_bg['cleaned_image']
    
    # Classificazione per ottenere la maschera (ignora il piedistallo)
    print("Running classify_beam per ottenere la maschera...")
    class_res = dl.classify_beam(cleaned_img)
    blob_mask = class_res.get('primary_mask')
    
    # Analisi Pointing Profile
    print("Running analyze_pointing_profile...")
    res = dl.analyze_pointing_profile(cleaned_img, blob_mask=blob_mask)
    
    x_c = res.get('X_c')
    y_c = res.get('Y_c')
    sigma_x = res.get('Sigma_X')
    sigma_y = res.get('Sigma_Y')
    
    print(f"Results: X_c={x_c:.1f}, Y_c={y_c:.1f}, Sigma_X={sigma_x:.1f}, Sigma_Y={sigma_y:.1f}")
    
    # Output directory
    out_dir = r"c:\Users\ILILUser\Desktop\Stability_Analysis\debug_output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Grafico 1: Immagine 2D con centroide e standard deviation
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cleaned_img, cmap='viridis', origin='upper')
    
    if x_c is not None and y_c is not None:
        ax.plot(x_c, y_c, 'r+', markersize=15, markeredgewidth=2, label=f'Centroid ({x_c:.1f}, {y_c:.1f})')
        # Disegna un'ellisse per rappresentare le Sigma (1 deviazione standard)
        from matplotlib.patches import Ellipse
        ellipse = Ellipse((x_c, y_c), width=2*sigma_x, height=2*sigma_y, 
                          edgecolor='red', facecolor='none', linestyle='--', linewidth=2, label=r'1$\sigma$ contour')
        ax.add_patch(ellipse)
        
    ax.set_title('Pointing Lanex - Shot 30')
    ax.legend()
    fig.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_pointing_2d.png"), dpi=150)
    plt.close(fig)
    print("Saved 03_pointing_2d.png")
    
    # Grafico 2: Profili Integrati X e Y
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Profilo integrato lungo X (somma lungo Y/righe)
    prof_x = np.sum(cleaned_img, axis=0)
    x_axis = np.arange(len(prof_x))
    axes[0].plot(x_axis, prof_x, 'b-', label='Integrated X Profile')
    if x_c is not None:
        axes[0].axvline(x_c, color='r', linestyle='--', label=f'X_c = {x_c:.1f}')
        axes[0].axvspan(x_c - sigma_x, x_c + sigma_x, color='r', alpha=0.2, label=r'$\pm 1\sigma_X$')
    axes[0].set_title('Spatial Profile X')
    axes[0].legend()
    
    # Profilo integrato lungo Y (somma lungo X/colonne)
    prof_y = np.sum(cleaned_img, axis=1)
    y_axis = np.arange(len(prof_y))
    axes[1].plot(y_axis, prof_y, 'g-', label='Integrated Y Profile')
    if y_c is not None:
        axes[1].axvline(y_c, color='r', linestyle='--', label=f'Y_c = {y_c:.1f}')
        axes[1].axvspan(y_c - sigma_y, y_c + sigma_y, color='r', alpha=0.2, label=r'$\pm 1\sigma_Y$')
    axes[1].set_title('Spatial Profile Y')
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_pointing_profiles.png"), dpi=150)
    plt.close(fig)
    print("Saved 04_pointing_profiles.png")
    
    print(f"Results saved in {out_dir}")

if __name__ == '__main__':
    main()
