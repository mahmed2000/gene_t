###########################################################################
#
#   Downloads mutations from ICGC's DCC search.
#
#   Requires:
#       Python packages: Xlib
#       Installed programs: wmctrl, xclip, any GUI browser
#
###########################################################################


import subprocess, time
import Xlib.X, Xlib.display, Xlib.ext.xtest, Xlib.XK

TOTAL = 609     # Total number of mutation occurences listed in search
PER_PAGE = 100  # Number of occurences to download per file (100 max)

# Browser to use. Name must be as displayed in output of wmctrl -l (Case sensitive).
# Must be open before running this script, Download JSON button on search page must be
# visible without scroling.
BROWSER = "Opera" 

# Search URL with filters. Edit location here, or copy from browser.
#
# If copying, use location range in mutations, instead of gene name or gene location
# After filtering, navigate to mutatioons -> occurences, show 50 rows. Paste here and
# edit "size":50,"from":1 -> "size":{},"from":{}
# Note, pipeline only handles single base mutations
URL = 'https://dcc.icgc.org/search/m/o?filters=%7B"mutation":%7B"location":%7B"is":%5B"17:7565097-7590856"%5D%7D,"type":%7B"is":%5B"single%20base%20substitution"%5D%7D%7D%7D&mutations=%7B"from":1%7D&occurrences=%7B"size":{},"from":{}%7D'

# Mouse cursor position when hovering over the JSON download button, set for 1920x1080
# To edit run in a shell, and hover over the download button for a few seconds:
# python -c 'import Xlib.display, time;time.sleep(4);print(Xlib.display.Display().screen().root.query_pointer())'
# Replace values with (root_x, root_y)
DL_POS = (1615, 827)

# Switches focus to BROWSER
subprocess.run(f"wmctrl -a {BROWSER}", shell = True)

time.sleep(1)


d = Xlib.display.Display()

for i in range(1, TOTAL, PER_PAGE):
    # Simulates Alt+d press. Jumps to address bar on most browsers.
    # If replacing, must select entire current address to be able to replace it entirely
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, d.keysym_to_keycode(Xlib.XK.XK_Alt_L))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, d.keysym_to_keycode(Xlib.XK.XK_d))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, d.keysym_to_keycode(Xlib.XK.XK_d))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, d.keysym_to_keycode(Xlib.XK.XK_Alt_L))
    d.sync()

    # Copies formatted url to clipboard)
    subprocess.run(f"echo '{URL.format(PER_PAGE, i)}' | xclip -sel clip", shell = True)
    time.sleep(0.5)

    # Simulates Ctrl+v.
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, d.keysym_to_keycode(Xlib.XK.XK_Control_L))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, d.keysym_to_keycode(Xlib.XK.XK_v))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, d.keysym_to_keycode(Xlib.XK.XK_v))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, d.keysym_to_keycode(Xlib.XK.XK_Control_L))
    d.sync()

    # Enter press
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, d.keysym_to_keycode(Xlib.XK.XK_Return))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, d.keysym_to_keycode(Xlib.XK.XK_Return))
    d.sync()

    # Wait for all elements on page to load
    time.sleep(15)

    # Moves mouse to button coords and clicks
    d.screen().root.warp_pointer(*DL_POS)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.ButtonPress, 1)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.ButtonRelease, 1)
    d.sync()

    # Wait for download to start
    time.sleep(1)


