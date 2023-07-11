from scipy.stats import ks_2samp
import scipy.stats as stats


def create_groups(df, x, y):
    temp_df = df[[x, y]]
    group0 = temp_df[y][temp_df[x] == 0].tolist()
    group1 = temp_df[y][temp_df[x] == 1].tolist()

    return group0, group1

def dist_diff(df, x, y, print = True):
    group0, group1 = create_groups(df, x, y)

    # Check difference between distributions
    dist_pval = ks_2samp(group0, group1).pvalue

    if print == True:
        if dist_pval < 0.05:
            print('Groups are not from the same distribution')
        else:
            print('Groups are from the same distribution')

    return dist_pval

def norm_check(df, x, y, print = True):
    group0, group1 = create_groups(df, x, y)

    # Check normality
    norm_pval0 = stats.kstest(group0, 'norm').pvalue
    norm_pval1 = stats.kstest(group1, 'norm').pvalue

    if print == True:
        for i, pval in enumerate([norm_pval0, norm_pval1]):
            if pval < 0.05:
                print('Group' + str(i) + ' is not from a normal distribution')
            else:
                print('Group' + str(i) + ' is from a normal distribution')

    return norm_pval0, norm_pval1

def get_wilcox_ttest(df, x, y):
    group0, group1 = create_groups(df, x, y)

    # Check normality
    norm_pval0, norm_pval1 = norm_check(df, x, y)

    # Check difference between distributions
    dist_pval = dist_diff(df, x, y)

    ## groups do not come from the same distribution or not normal
    if norm_pval0 < 0.05 or norm_pval1 < 0.05 or dist_pval < 0.05:
        p_val = stats.mannwhitneyu(group0, group1)

    ## groups come from the same distribution and normal
    else:
        p_val = stats.ttest_ind(group0, group1)
    print(p_val)

    return p_val[1]