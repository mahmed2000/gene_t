import subprocess, time
import Xlib.X, Xlib.display, Xlib.ext.xtest, Xlib.XK

TOTAL = 609
PER_PAGE = 100
BROWSER = "Opera"

URL = 'https://dcc.icgc.org/search/m/o?filters=%7B"mutation":%7B"location":%7B"is":%5B"17:7565097-7590856"%5D%7D,"type":%7B"is":%5B"single%20base%20substitution"%5D%7D%7D%7D&mutations=%7B"from":1%7D&occurrences=%7B"size":{},"from":{}%7D'

DL_POS = (1615, 827)

subprocess.run(f"wmctrl -a {BROWSER}", shell = True)

time.sleep(1)

d = Xlib.display.Display()



for i in range(1,TOTAL,PER_PAGE):
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, d.keysym_to_keycode(Xlib.XK.XK_Alt_L))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, d.keysym_to_keycode(Xlib.XK.XK_d))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, d.keysym_to_keycode(Xlib.XK.XK_d))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, d.keysym_to_keycode(Xlib.XK.XK_Alt_L))
    d.sync()

    subprocess.run(f"echo '{URL.format(PER_PAGE, i)}' | xclip -sel clip", shell = True)
    time.sleep(0.5)

    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, d.keysym_to_keycode(Xlib.XK.XK_Control_L))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, d.keysym_to_keycode(Xlib.XK.XK_v))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, d.keysym_to_keycode(Xlib.XK.XK_v))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, d.keysym_to_keycode(Xlib.XK.XK_Control_L))
    d.sync()

    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, d.keysym_to_keycode(Xlib.XK.XK_Return))
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, d.keysym_to_keycode(Xlib.XK.XK_Return))
    d.sync()

    time.sleep(15)

    d.screen().root.warp_pointer(*DL_POS)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.ButtonPress, 1)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.ButtonRelease, 1)
    d.sync()

    time.sleep(1)


