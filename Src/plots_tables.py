import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, classification_report,\
                            ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import six

def print_report(y_true, y_pred, model_name=None, display_labels=None):

    print("\n")
    print(" BALANCED ACCURACY: ",balanced_accuracy_score(y_true=y_true, 
                                                        y_pred=y_pred))
    print("\n")
    print("\n")
    print("CLASSIFICATION REPORT:")
    print("\n")
    print(classification_report(y_true=y_true, 
                                y_pred=y_pred, 
                                target_names=display_labels))

    fig, axs = plt.subplots(nrows=1, ncols=2, 
                            figsize=(25, 10),
                            sharex=True)

    print("\n")
    print("\n")
    __confusion_matrix__ = confusion_matrix(y_true=y_true, 
                                            y_pred=y_pred)

    ax = axs[0]
    ConfusionMatrixDisplay(__confusion_matrix__,display_labels=display_labels).plot(cmap='Blues',values_format='d',ax=ax)
    ax.set_title('Non Normalized',fontsize=20)


    __confusion_matrix__ = confusion_matrix(y_true=y_true, 
                                        y_pred=y_pred,
                                        normalize='true')

    ax = axs[1]
    ConfusionMatrixDisplay(__confusion_matrix__,display_labels=display_labels).plot(cmap='Blues',values_format='.5f',ax=ax)
    ax.set_title('Normalized',fontsize=20)
                            
    fig.suptitle('True-Predicted labels Heat Map',fontsize=30)

    plt.savefig('tables/{}.png'.format(model_name))



def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        _, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax