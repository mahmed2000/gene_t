import i3ipc, subprocess, time
import Xlib.X, Xlib.display, Xlib.ext.xtest, Xlib.XK

TOTAL = 609
PER_PAGE = 100

URL = 'https://dcc.icgc.org/search/m/o?filters=%7B"gene":%7B"id":%7B"is":%5B"ENSG00000141510"%5D%7D%7D,"donor":%7B"primarySite":%7B"is":%5B"Breast"%5D%7D%7D,"mutation":%7B"type":%7B"is":%5B"single%20base%20substitution"%5D%7D%7D%7D&donors=%7B"from":1%7D&mutations=%7B"from":1%7D&occurrences=%7B"size":{},"from":{}%7D'

DL_POS = (1615, 827)


i3 = i3ipc.Connection()
i3.command('workspace 3')

time.sleep(1)

d = Xlib.display.Display()

for i in range(1,TOTAL,PER_PAGE):
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, 37)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, 26)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, 26)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, 37)
    d.sync()

    subprocess.run(f"echo '{URL.format(PER_PAGE, i)}' | xclip -sel clip", shell = True)
    time.sleep(0.5)

    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, 37)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, 55)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, 55)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, 37)
    d.sync()

    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyPress, 36)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.KeyRelease, 36)
    d.sync()

    time.sleep(15)

    d.screen().root.warp_pointer(*DL_POS)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.ButtonPress, 1)
    d.sync()
    Xlib.ext.xtest.fake_input(d, Xlib.X.ButtonRelease, 1)
    d.sync()

    time.sleep(1)

i3.command('workspace 1')
