from pymol.cgo import *
obj = [BEGIN,LINES,COLOR,1.00,1.00,1.00,VERTEX,-61.544510771831824,-51.69286250974942,-57.16483398010564,VERTEX,58.45548922816817,-51.69286250974942,-57.16483398010564,VERTEX,58.45548922816817,-51.69286250974942,-57.16483398010564,VERTEX,58.45548922816817,53.30713749025058,-57.16483398010564,VERTEX,58.45548922816817,-51.69286250974942,-57.16483398010564,VERTEX,58.45548922816817,-51.69286250974942,62.83516601989436,VERTEX,-61.544510771831824,-51.69286250974942,-57.16483398010564,VERTEX,-61.544510771831824,53.30713749025058,-57.16483398010564,VERTEX,-61.544510771831824,53.30713749025058,-57.16483398010564,VERTEX,58.45548922816817,53.30713749025058,-57.16483398010564,VERTEX,-61.544510771831824,53.30713749025058,-57.16483398010564,VERTEX,-61.544510771831824,53.30713749025058,62.83516601989436,VERTEX,-61.544510771831824,-51.69286250974942,-57.16483398010564,VERTEX,-61.544510771831824,-51.69286250974942,62.83516601989436,VERTEX,-61.544510771831824,-51.69286250974942,62.83516601989436,VERTEX,58.45548922816817,-51.69286250974942,62.83516601989436,VERTEX,-61.544510771831824,-51.69286250974942,62.83516601989436,VERTEX,-61.544510771831824,53.30713749025058,62.83516601989436,VERTEX,-61.544510771831824,53.30713749025058,62.83516601989436,VERTEX,58.45548922816817,53.30713749025058,62.83516601989436,VERTEX,58.45548922816817,-51.69286250974942,62.83516601989436,VERTEX,58.45548922816817,53.30713749025058,62.83516601989436,VERTEX,58.45548922816817,53.30713749025058,-57.16483398010564,VERTEX,58.45548922816817,53.30713749025058,62.83516601989436,END]
cmd.load_cgo(obj, "max_system_box")