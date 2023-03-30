import os
import sys
import pandas as pd
import numpy as np 
import requests

from io import BytesIO
from glob import glob
from PIL import Image, ImageEnhance

import streamlit as st

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageColor
import colorsys
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans, BisectingKMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE


model_dict = {
    "KMeans": KMeans,
    "BisectingKMeans" : BisectingKMeans,
    "GaussianMixture": GaussianMixture,
    "MiniBatchKMeans": MiniBatchKMeans,
}

center_method = {
    "KMeans": "cluster_centers_",
    "BisectingKMeans" : "cluster_centers_",
    "GaussianMixture": "means_",
    "MiniBatchKMeans": "cluster_centers_",
}

n_cluster_arg = {
    "KMeans": "n_clusters",
    "BisectingKMeans" : "n_clusters",
    "GaussianMixture": "n_components",
    "MiniBatchKMeans": "n_clusters",

}

enhancement_range = {
    "Color": [0., 5., 0.2], 
    "Sharpness": [0., 3., 0.2], 
    "Contrast": [0.5, 1.5, 0.1], 
    "Brightness": [0.5, 1.5, 0.1]
}

sort_func_dict = {
    "rgb": (lambda r,g,b: (r, g, b)),
    "sum_rgb": (lambda r,g,b: r+g+b),
    "sqr_rgb": (lambda r,g,b: r**2+g**2+b**2),
    "hsv": (lambda r, g, b : colorsys.rgb_to_hsv(r, g, b)),
    "random": (lambda r, g, b: np.random.random())
}

def get_df_rgb(img, sample_size):
    """construct a sample RGB dataframe from image"""

    n_dims = np.array(img).shape[-1]
    r,g,b = np.array(img).reshape(-1,n_dims).T
    df = pd.DataFrame({"R": r, "G": g, "B": b}).sample(n=sample_size)
    return df

def get_palette(df_rgb, model_name, palette_size, sort_func="random"):
    """cluster pixels together and return a sorted color palette."""
    params = {n_cluster_arg[model_name]: palette_size}
    model = model_dict[model_name](**params)

    clusters = model.fit_predict(df_rgb)
        
    palette = getattr(model, center_method[model_name]).astype(int).tolist()
    
    palette.sort(key=lambda rgb : sort_func_dict[sort_func.rstrip("_r")](*rgb), 
                reverse=bool(sort_func.endswith("_r")))

    return palette

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def show_palette(palette_hex):
    """show palette strip"""
    palette = np.array([ImageColor.getcolor(color, "RGB") for color in  palette_hex])
    fig, ax = plt.subplots(dpi=100)
    ax.imshow(palette[np.newaxis, :, :])
    ax.axis('off')
    return fig


def store_palette(palette):
    """store palette colors in session state"""
    palette_size = len(palette)
    columns = st.columns(palette_size)
    for i, col in enumerate(columns):
        with col:        
            st.session_state[f"col_{i}"]= st.color_picker(label=str(i), value=rgb_to_hex(palette[i]), key=f"pal_{i}")

def display_matplotlib_code(palette_hex):

    st.write('Use this snippet in your code to make your color palette more sophisticated!')
    code = st.code(f"""
import matplotlib as mpl
from cycler import cycler

palette = {palette_hex}
mpl.rcParams["axes.prop_cycle"] = cycler(color=palette)
    """
    )   

def display_plotly_code(palette_hex):
    st.write('Use this snippet in your code to make your color palette more sophisticated!')
    st.code(f"""
import plotly.io as pio
import plotly.graph_objects as go
pio.templates["sophisticated"] = go.layout.Template(
    layout=go.Layout(
    colorway={palette_hex}
    )
)
pio.templates.default = 'sophisticated'
            """)

def plot_rgb_3d(df_rgb):
    """plot the sampled pixels in 3D RGB space"""

    if df_rgb.shape[0] > 2000:
        st.error("RGB plot can only be used for less than 2000 sample pixels.")
    else:
        colors = df_rgb.apply(rgb_to_hex, axis=1)
        fig = px.scatter_3d(df_rgb, x='R', y='G', z='B',
                color=colors, size=[1]*df_rgb.shape[0],
                opacity=0.7)

        st.plotly_chart(fig)


def plot_hsv_3d(df):
    """plot the sampled pixels in 3D RGB space"""
    df_rgb = df.copy()
    if df_rgb.shape[0] > 2000:
        st.error("RGB plot can only be used for less than 2000 sample pixels.")

    else:
        df_rgb[["H","S",'V']]= df_rgb.apply(lambda x: pd.Series(colorsys.rgb_to_hsv(x.R/255.,x.G/255.,x.B/255.)).T, axis=1)
        st.dataframe(df_rgb[["H","S",'V']])
        colors = df_rgb[["R","G","B"]].apply(rgb_to_hex, axis=1)
        fig = px.scatter_3d(df_rgb, x='H', y='S', z='V',
                color=colors, size=[1]*df_rgb.shape[0],
                opacity=0.7)

        st.plotly_chart(fig)

def print_praise():
    """Yes, I'm that vain and superficial! 🙄 """

    praise_quotes = [
        '"When I stumbled upon this app, it was like I found a *pearl* among the oysetrs. Absolutely stunning! "\n\n-- Johannes Merveer',
        '"I wish *Mona* was alive to see this masterpiece! I\'m sure she would have *smiled* at it..."\n\n-- Leonarda va Dinci',
        '"I\'m sorry, what was that? Ah yes, great app. I use it every *night*. Five *stars*!"\n\n-- Vincent van Vogue',
        '"We\'ve all been waiting years for an app to make a *big splash* like this, and now it\'s finally here!\n[Can you hand me that towel please?]"\n\n-- David Hockknee',
        '"It makes such a great *impression* on you, doesn\'t it? I know where I\'ll be getting my palette for painting the next *sunrise*!"\n\n-- Cloud Moanet',
        '"Maybe some other time... [Can I get a gin and tonic please?]"\n\n-- Edward Jumper',
    ]

    title = "[imaginary] **Praise for Sophisticated Palette**\n\n"
    # random_index = np.random.randint(len(praise_quotes))
    weights = np.array([2, 3.5, 3, 3, 3, 1])
    weights = weights/np.sum(weights)

    return title + np.random.choice(praise_quotes, p=weights)


gallery_files = glob(os.path.join(".", "images", "*"))
gallery_dict = {image_path.split("/")[-1].split(".")[-2].replace("-", " "): image_path
    for image_path in gallery_files}







st.image("logo.jpg")
st.sidebar.title("CreditCard Fraud Detection 🎨")
st.sidebar.caption("Test your models for different Testset.")
# st.sidebar.markdown("Made by [Siavash Yasini](https://www.linkedin.com/in/siavash-yasini/)")
# st.sidebar.caption("Look for the source Code [here](https://github.com/Keval78/Credit-Card-Fraud-Detection).")
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.header("Settings")
palette_size = int(st.sidebar.number_input("palette size", min_value=1, max_value=20, value=5, step=1, help="Number of colors to infer from the image."))
sample_size = int(st.sidebar.number_input("sample size", min_value=5, max_value=3000, value=500, step=500, help="Number of sample pixels to pick from the image."))

# Image Enhancement
enhancement_categories = enhancement_range.keys()
enh_expander = st.sidebar.expander("Image Enhancements", expanded=False)
with enh_expander:
    
    if st.button("reset"):
        for cat in enhancement_categories:
            if f"{cat}_enhancement" in st.session_state:
                st.session_state[f"{cat}_enhancement"] = 1.0
enhancement_factor_dict = {
    cat: enh_expander.slider(f"{cat} Enhancement", 
                            value=1., 
                            min_value=enhancement_range[cat][0], 
                            max_value=enhancement_range[cat][1], 
                            step=enhancement_range[cat][2],
                            key=f"{cat}_enhancement")
    for cat in enhancement_categories
}
enh_expander.info("**Try the following**\n\nColor Enhancements = 2.6\n\nContrast Enhancements = 1.1\n\nBrightness Enhancements = 1.1")

# Clustering Model 
model_name = st.sidebar.selectbox("machine learning model", model_dict.keys(), help="Machine Learning model to use for clustering pixels and colors together.")
sklearn_info = st.sidebar.empty()

sort_options = sorted(list(sort_func_dict.keys()) + [key + "_r" for key in sort_func_dict.keys() if key!="random"])
sort_func = st.sidebar.selectbox("palette sort function", options=sort_options, index=5)

# Random Number Seed
seed = int(st.sidebar.number_input("random seed", value=42, help="Seed used for all random samplings."))
np.random.seed(seed)
st.sidebar.markdown("---")
st.sidebar.markdown("---")





# =======
#   App
# =======

# provide options to either select an image form the gallery, upload one, or fetch from URL
gallery_tab, upload_tab, url_tab = st.tabs(["Gallery", "Upload", "Image URL"])
with gallery_tab:
    options = list(gallery_dict.keys())
    file_name = st.selectbox("Select Art", 
                            options=options, index=options.index("Mona Lisa (Leonardo da Vinci)"))
    file = gallery_dict[file_name]

    if st.session_state.get("file_uploader") is not None:
        st.warning("To use the Gallery, remove the uploaded image first.")
    if st.session_state.get("image_url") not in ["", None]:
        st.warning("To use the Gallery, remove the image URL first.")

    img = Image.open(file)

with upload_tab:
    file = st.file_uploader("Upload Art", key="file_uploader")
    if file is not None:
        try:
            img = Image.open(file)
        except:
            st.error("The file you uploaded does not seem to be a valid image. Try uploading a png or jpg file.")
    if st.session_state.get("image_url") not in ["", None]:
        st.warning("To use the file uploader, remove the image URL first.")

with url_tab:
    url_text = st.empty()
    
    # FIXME: the button is a bit buggy, but it's worth fixing this later

    # url_reset = st.button("Clear URL", key="url_reset")
    # if url_reset and "image_url" in st.session_state:
    #     st.session_state["image_url"] = ""
    #     st.write(st.session_state["image_url"])

    url = url_text.text_input("Image URL", key="image_url")
    
    if url!="":
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
        except:
            st.error("The URL does not seem to be valid.")

# convert RGBA to RGB if necessary
n_dims = np.array(img).shape[-1]
if n_dims == 4:
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
    img = background

# apply image enhancements
for cat in enhancement_categories:
    img = getattr(ImageEnhance, cat)(img)
    img = img.enhance(enhancement_factor_dict[cat])

# show the image
with st.expander("🖼  Artwork", expanded=True):
    st.image(img, use_column_width=True)



st.info("👈  Click on 'Find Palette' ot turn on 'Toggle Update' to see the color palette.")

st.sidebar.success(print_praise())   
st.sidebar.write("---\n")
st.sidebar.caption("""You can check out the source code [here](https://github.com/syasini/sophisticated_palette).
                      The `matplotlib` and `plotly` code snippets have been borrowed from [here](https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html) and [here](https://stackoverflow.com/questions/63011674/plotly-how-to-change-the-default-color-pallete-in-plotly).""")
st.sidebar.write("---\n")
