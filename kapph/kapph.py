#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r runtime %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

class preProcessing(object):
    
    def demo_print():
        print("Hi user..")
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.cross_validation import train_test_split
     
    featureScaling = True
    dataSplit = True
    splitRatio = 0.2
    
    def __init__(self):
        """ Constructor Class for preprocessing """
        
    
    def dataArray(self,data, yName = False):
        if yName == False:
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
        else:
            y = data.iloc[:, data.columns == yName]
            X = data.iloc[:, data.columns != yName]
            
        return (X, y)

    def splitter(self,X,y,SR= splitRatio):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = SR, random_state = 0)
        return X_train, X_test, y_train, y_test
    
    @timeit
    def printCols(self,data):
        #print("The number of NA values in the column")
        #print(data.isna().sum())
        objCol = list(data.select_dtypes(include = ['object']).columns)
        numCol = list(data.select_dtypes(include = ['float64','int64']).columns)
        columndetails = []
        for i in objCol:
            columndetails.append({'Column Name':i,'Type' : 'Object' ,'Number of NULL values': float(data[i].isna().sum())})
        for i in numCol:
            columndetails.append({'Column Name':i,'Type' : 'Numeric' ,'Number of NULL values': float(data[i].isna().sum())})
        return(pd.DataFrame(columndetails))
        
        
    @timeit    
    def convertToObj(self,data,colToCon="all"):
        if colToCon == "all":
            for col in data.columns:
                data['col']  = data['col'].astype('object') 
        else:
            for col in colToCon:
                data[data.columns[col-1]] = data[data.columns[col-1]].astype('object')

    def convertToNum(self,data,colToCon="all"):
        if colToCon == "all":
            for col in data.columns:
                data['col']  = data['col'].astype('float64') 
        else:
            for col in colToCon:
                data[data.columns[col-1]] = data[data.columns[col-1]].astype('float64')
                
    def binning(self,data,col,valueList,labelNames):
        data[col] = pd.cut(data[col],valueList,labels = labelNames)
        data[col] = data[col].astype('object')
        return data
        
        
        
    def removeNull(self,data):
        nullCount = data.isnull().sum()
        data = data.dropna()
        return nullCount
    
    def oneHotEncoding(self,data):
        data = pd.get_dummies(data,drop_first = True)
        return data
    
        
        
########################################################

    
    def residual_plot(model_fit,df):
        
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        import statsmodels.formula.api as smf
        from statsmodels.graphics.gofplots import ProbPlot 
        import statsmodels.api as sm
        # Required calculation for the plot:
            
        # fitted values (need a constant term for intercept)
        model_fitted_y = model_fit.fittedvalues
        
        # model residuals
        model_residuals = model_fit.resid
        
        # normalized residuals
        model_norm_residuals = model_fit.get_influence().resid_studentized_internal
        
        # absolute squared normalized residuals
        model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
        
        # absolute residuals
        model_abs_resid = np.abs(model_residuals)
        
        # leverage, from statsmodels internals
        model_leverage = model_fit.get_influence().hat_matrix_diag
        
        # cook's distance, from statsmodels internals
        model_cooks = model_fit.get_influence().cooks_distance[0]
        
        ## Residual plot
    
        plot_lm_1 = plt.figure(1)
        plot_lm_1.set_figheight(8)
        plot_lm_1.set_figwidth(12)
        
        plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'Time', data=df, 
                                  lowess=True, 
                                  scatter_kws={'alpha': 0.5}, 
                                  line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        
        plot_lm_1.axes[0].set_title('Residuals vs Fitted')
        plot_lm_1.axes[0].set_xlabel('Fitted values')
        plot_lm_1.axes[0].set_ylabel('Residuals')
        
        # annotations
        abs_resid = model_abs_resid.sort_values(ascending=False)
        abs_resid_top_3 = abs_resid[:3]
        
        for i in abs_resid_top_3.index:
            plot_lm_1.axes[0].annotate(i, 
                                       xy=(model_fitted_y[i], 
                                           model_residuals[i]));
        
                          
        ## Q-Q plot
        
        QQ = ProbPlot(model_norm_residuals)
        plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
        
        plot_lm_2.set_figheight(8)
        plot_lm_2.set_figwidth(12)
        
        plot_lm_2.axes[0].set_title('Normal Q-Q')
        plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
        plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
        
        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        
        for r, i in enumerate(abs_norm_resid_top_3):
            plot_lm_2.axes[0].annotate(i, 
                                       xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                           model_norm_residuals[i]));
                          
        ## Scale Location Plot
    
        plot_lm_3 = plt.figure(3)
        plot_lm_3.set_figheight(8)
        plot_lm_3.set_figwidth(12)
        
        plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
        sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
                    scatter=False, 
                    ci=False, 
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        
        plot_lm_3.axes[0].set_title('Scale-Location')
        plot_lm_3.axes[0].set_xlabel('Fitted values')
        plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');
        
        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        
        for i in abs_norm_resid_top_3:
            plot_lm_3.axes[0].annotate(i, 
                                       xy=(model_fitted_y[i], 
                                           model_norm_residuals_abs_sqrt[i]));
                          
        ## Leverage plot
        
        plot_lm_4 = plt.figure(4)
        plot_lm_4.set_figheight(8)
        plot_lm_4.set_figwidth(12)
        
        plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
        sns.regplot(model_leverage, model_norm_residuals, 
                    scatter=False, 
                    ci=False, 
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        
        plot_lm_4.axes[0].set_xlim(0, 0.20)
        plot_lm_4.axes[0].set_ylim(-3, 5)
        plot_lm_4.axes[0].set_title('Residuals vs Leverage')
        plot_lm_4.axes[0].set_xlabel('Leverage')
        plot_lm_4.axes[0].set_ylabel('Standardized Residuals')
        
        # annotations
        leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
        
        for i in leverage_top_3:
            plot_lm_4.axes[0].annotate(i, 
                                       xy=(model_leverage[i], 
                                           model_norm_residuals[i]))
            
        # shenanigans for cook's distance contours
        def graph(formula, x_range, label=None):
            x = x_range
            y = formula(x)
            plt.plot(x, y, label=label, lw=1, ls='--', color='red')
        
        p = len(model_fit.params) # number of model parameters
        
        graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
              np.linspace(0.001, 0.200, 50), 
              'Cook\'s distance') # 0.5 line
        graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
              np.linspace(0.001, 0.200, 50)) # 1 line
        plt.legend(loc='upper right');
        
        return True