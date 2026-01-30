import matplotlib.font_manager as fm
import os


# Find the specific font file for Noto Sans CJK JP
font_list = fm.findSystemFonts()
cjk_font_path = [f for f in font_list if 'NotoSansCJK' in f or 'NotoSansCJKJP' in f]
print(f"Font paths found: {cjk_font_path}")

# Set the font properties using the path
if cjk_font_path:
    my_font = fm.FontProperties(fname=cjk_font_path[0])
    print(f"Using font: {my_font.get_name()}")
else:
    print("No CJK font path found.")


    