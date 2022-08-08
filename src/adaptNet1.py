from wn_util import net1
from time_constants import SECONDS_PER_DAY

wn = net1()
wn.options.time.duration = 7 * SECONDS_PER_DAY
wn.write_inpfile('Net1_adapted.inp')
