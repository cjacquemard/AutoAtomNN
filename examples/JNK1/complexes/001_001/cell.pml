from pymol.cgo import *
obj = [BEGIN,LINES,COLOR,0.00,1.00,0.00,VERTEX,-27.218826293945312,-31.0997371673584,-31.363889694213867,VERTEX,27.05421257019043,-31.0997371673584,-31.363889694213867,VERTEX,27.05421257019043,-31.0997371673584,-31.363889694213867,VERTEX,27.05421257019043,34.84049606323242,-31.363889694213867,VERTEX,27.05421257019043,-31.0997371673584,-31.363889694213867,VERTEX,27.05421257019043,-31.0997371673584,29.893522262573242,VERTEX,-27.218826293945312,-31.0997371673584,-31.363889694213867,VERTEX,-27.218826293945312,34.84049606323242,-31.363889694213867,VERTEX,-27.218826293945312,34.84049606323242,-31.363889694213867,VERTEX,27.05421257019043,34.84049606323242,-31.363889694213867,VERTEX,-27.218826293945312,34.84049606323242,-31.363889694213867,VERTEX,-27.218826293945312,34.84049606323242,29.893522262573242,VERTEX,-27.218826293945312,-31.0997371673584,-31.363889694213867,VERTEX,-27.218826293945312,-31.0997371673584,29.893522262573242,VERTEX,-27.218826293945312,-31.0997371673584,29.893522262573242,VERTEX,27.05421257019043,-31.0997371673584,29.893522262573242,VERTEX,-27.218826293945312,-31.0997371673584,29.893522262573242,VERTEX,-27.218826293945312,34.84049606323242,29.893522262573242,VERTEX,-27.218826293945312,34.84049606323242,29.893522262573242,VERTEX,27.05421257019043,34.84049606323242,29.893522262573242,VERTEX,27.05421257019043,-31.0997371673584,29.893522262573242,VERTEX,27.05421257019043,34.84049606323242,29.893522262573242,VERTEX,27.05421257019043,34.84049606323242,-31.363889694213867,VERTEX,27.05421257019043,34.84049606323242,29.893522262573242,END]
cmd.load_cgo(obj, "cell")