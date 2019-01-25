import ImportData
import Details as D
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Process, Pool
import multiprocessing.managers
import Maths

area = 'ACC'
area = 'FP'
epoch = D.sc_secondaryreinforceron
epoch = D.sc_transition
num_x = 4


data = ImportData.EntireArea(area)
a=data.behavdata[0]

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


def analysecell(counter, n_counter, ar_repeat_commons, ar_repeat_rares, ar_switch_commons, ar_switch_rares, cell):
    counter[cell] = 1
    print(f'{int(sum(counter))}/{data.n}')
    frpertrial = data.generateaverageepoch(cell, epoch)
    trialdata = data.behavdata[cell]

    # Lets reproduce Fig 5.14a left panel
    trialtype = trialdata.trialtype
    prevoutcome = trialdata.previousreward
    transition = trialdata.transition
    repeatc2atc1 = trialdata.repeatc2atc1

    # Split into common and rare
    fr_repeat_common = frpertrial[(trialtype==1) & (prevoutcome==1) & (repeatc2atc1==1) & (transition==1) ]
    fr_repeat_rare = frpertrial[(trialtype==1) & (prevoutcome==1) & (repeatc2atc1==1) & (transition==2)]
    fr_switch_common = frpertrial[(trialtype==1) & (prevoutcome==1) & (repeatc2atc1==0) & (transition==1)]
    fr_switch_rare = frpertrial[(trialtype==1) & (prevoutcome==1) & (repeatc2atc1==0) & (transition==2)]

    # Average data
    ar_repeat_commons[cell] = Maths.normalise(np.mean(fr_repeat_common, axis=0))
    ar_repeat_rares[cell] = Maths.normalise(np.mean(fr_repeat_rare, axis=0))
    ar_switch_commons[cell] = Maths.normalise(np.mean(fr_switch_common, axis=0))
    ar_switch_rares[cell] = Maths.normalise(np.mean(fr_switch_rare, axis=0))

    def updatecounter(i, ar):
        if ar.shape[0] > 0:
            n_counter[i] += 1

    updatecounter(0, fr_repeat_common)
    updatecounter(1, fr_repeat_rare)
    updatecounter(2, fr_switch_common)
    updatecounter(3, fr_switch_rare)


if __name__ == "__main__":
    m = MyManager()
    m.start()
    fr_repeat_commons = m.np_zeros((data.n, D.smooth_outputlength))
    fr_repeat_rares = m.np_zeros((data.n, D.smooth_outputlength))
    fr_switch_commons = m.np_zeros((data.n, D.smooth_outputlength))
    fr_switch_rares = m.np_zeros((data.n, D.smooth_outputlength))
    counter = m.np_zeros(data.n)
    x_counter = m.np_zeros(num_x)

    pool = Pool()
    func = partial(analysecell, counter, x_counter, fr_repeat_commons, fr_repeat_rares, fr_switch_commons, fr_switch_rares)
    run_list = range(data.n)
    pool.map(func, run_list)  # Now put run_list in the second argument of local_func
    pool.close()

    fr_repeat_commons = np.array(fr_repeat_commons)
    fr_repeat_rares = np.array(fr_repeat_rares)
    fr_switch_commons = np.array(fr_switch_commons)
    fr_switch_rares = np.array(fr_switch_rares)
    x_counter = np.array(x_counter)

    for arr, label in zip((fr_repeat_commons, fr_repeat_rares, fr_switch_commons, fr_switch_rares), ('Repeat Common', 'Repeat rare', 'Switch common', 'Switch rare')):
        plt.plot(np.nanmean(arr, axis=0), label=label)
    plt.legend()
    plt.show()





