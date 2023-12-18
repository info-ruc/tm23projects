import numpy as np
import pandas
from PyEMD import EMD, EEMD, Visualisation  # Visualisation 可能缺引numpy
from scipy import stats


def load_data_to_signal(filename):   
    """load the dataset

    Disc：
        get dataframe's values, transfer to (1, n)
    """
    dataframe = pandas.read_csv(filename, engine='python')
    dataframe = dataframe['price']
    dataset = dataframe.values   # Return a Numpy representation of the DataFrame
    dataset = dataset.astype('float32')  # 变为 float type
    dataset = np.reshape(dataset, (dataset.shape[0]),)  
    return dataset


def fine_to_coarse(imfs):
    """高频-低频重构（Fine-to-coarse Reconstruction）

    Disc：
        imfs: without residue
        return: imf_fine, imf_coarse, dataframe for t test, index
    """
    imf_sum = np.zeros(imfs.shape[1])
    imf_fine = np.zeros(imfs.shape[1])
    imf_coarse = np.zeros(imfs.shape[1])
    df = pandas.DataFrame(index=['t', 'p'], columns=np.arange(1, len(imfs)+1))
    for n, imf in enumerate(imfs):
        imf_sum += imf
        tt, pval = stats.ttest_1samp(imf_sum, 0.0)
        # print('t-statistic = %6.3f pvalue = %6.4f' % (tt, pval))
        df.iloc[0, n] = tt
        df.iloc[1, n] = pval
        if(pval < 0.05):
            index = n  # 低频从该处起
            # print("n=%d" % (n+1))
            imf_fine = imf_sum - imf
            imf_coarse += imf
            break
    for n in range(index+1, len(imfs)):
        imf_sum += imfs[n]
        tt, pval = stats.ttest_1samp(imf_sum, 0.0)
        # print('t-statistic = %6.3f pvalue = %6.4f' % (tt, pval))
        df.iloc[0, n] = tt
        df.iloc[1, n] = pval
        imf_coarse += imfs[n]
    return imf_fine, imf_coarse, df, index+1


def singal_to_supervised(singal):
    """singal_to_supervised

    Disc：
        singal: univariate series 
        return: 
            _multi: (singal, imf_fine, imf_coarse, res)
            vis_de, vis_re
    """
    # Extract imfs and residue， In case of EEMD
    eemd = EEMD()
    imfs = eemd(singal)
    res = imfs[-1, :]
    imfs= imfs[0:-1, :]
    
    vis_de = Visualisation()
    vis_de.plot_imfs(imfs=imfs, residue=res, include_residue=True)
    # vis.plot_instant_freq(t, imfs=imfs)
    
    imf_fine, imf_coarse, df, index = fine_to_coarse(imfs)
    print("# index: ", index)
    imf_re = np.vstack((imf_fine, imf_coarse))
    vis_re = Visualisation()
    vis_re.plot_imfs(imfs=imf_re, residue=res, include_residue=True)
    
    # test fine_to_coarse(imfs)
    diff = singal - imf_fine - imf_coarse - res
    print("test fine_to_coarse(imfs)\n", diff)

    # tranfer dataset
    _multi = np.vstack((singal, imf_fine, imf_coarse, res))  # plus original price as feature and label
    _multi = _multi.T

    return _multi, vis_de, vis_re, df
