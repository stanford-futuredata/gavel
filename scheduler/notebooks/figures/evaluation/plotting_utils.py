# Matplotlib imports.
from matplotlib import pyplot as plt
import matplotlib; matplotlib.font_manager._rebuild()
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from pylab import *
import seaborn as sns
from matplotlib import rc
sns.set_style('ticks')
font = {
    'font.family':'Roboto',
    'font.size': 12,
}
sns.set_style(font)
paper_rc = {
    'lines.linewidth': 3,
    'lines.markersize': 10,
}
sns.set_context("paper", font_scale=2, rc=paper_rc)
current_palette = sns.color_palette()

# Other imports.
import os
import pandas as pd

def plot_metric_vs_inverse_lambda(logfile_paths,
                                  labels,
                                  v100s, p100s, k80s,
                                  policies, metric_fn,
                                  metric_label,
                                  xmax=None,
                                  ymax=None,
                                  output_filename=None,
                                  extrapolate=False,
                                  verbose=False):
    from utils import prune

    plt.figure(figsize=(8, 3))
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

    data = {"input_job_rate": [], "metric": [], "seed": [],
            "policy": []}
    print(policies)
    for policy in policies:
        relevant_logfile_paths = list(reversed(prune(
            logfile_paths, v100s, p100s, k80s, policy)))
        lambdas = [x[0] for x in relevant_logfile_paths]
        input_job_rates = [3600.0 / x for x in lambdas]
        metrics = [metric_fn(x[1]) for x in relevant_logfile_paths]
        seeds = [x[2] for x in relevant_logfile_paths]
        policies = [labels[policy] for i in range(len(metrics))]

        data["input_job_rate"] += input_job_rates
        data["metric"] += metrics
        data["seed"] += seeds
        data["policy"] += policies
        if len(input_job_rates) > 0 and extrapolate:
            data["input_job_rate"] += [max(input_job_rates) + 0.4]
            data["metric"] += [105.0]
            data["seed"] += [0]
            data["policy"] += [labels[policy]]
    if verbose:
        df = pd.DataFrame(data)
        grouped_df = df.groupby(["policy", "input_job_rate", "seed"])
        for name_of_the_group, group in grouped_df:
            print(name_of_the_group)
            print(group.mean())

    sns.lineplot(x='input_job_rate', y='metric', style='policy',
                 hue='policy',
                 data=data, ci='sd',
                 markers=True)

    ax.set_xlabel("Input job rate (jobs/hr)")
    ax.set_ylabel(metric_label)
    ax.set_xlim([0, xmax])
    ax.set_ylim([0, ymax])
    sns.despine()
    
    leg = plt.legend(loc='upper left', frameon=False)
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    bb.y0 += 0.22
    bb.y1 += 0.22
    leg.set_bbox_to_anchor(bb, transform=ax.transAxes)

    if output_filename is not None:
        with PdfPages(output_filename) as pdf:
            pdf.savefig(bbox_inches='tight')
    
    plt.show()


def plot_metric_vs_inverse_lambda_different_mechanisms(all_logfile_paths,
                                                       labels,
                                                       label_modifiers,
                                                       v100s, p100s, k80s,
                                                       policies, metric_fn,
                                                       metric_label,
                                                       xmax=None,
                                                       ymax=None,
                                                       output_filename=None,
                                                       extrapolate=False):
    from utils import prune

    plt.figure(figsize=(4.5, 3))
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

    data = {"input_job_rate": [], "metric": [], "seed": [],
            "policy": []}
    print(policies)
    for policy in policies:
        for logfile_paths, label_modifier in zip(
            all_logfile_paths, label_modifiers):
            relevant_logfile_paths = list(reversed(prune(
                logfile_paths, v100s, p100s, k80s, policy)))
            label = labels[policy] + label_modifier
            
            lambdas = []
            metrics = []
            seeds = []
            for x in relevant_logfile_paths:
                metric = metric_fn(x[1])
                if metric is not None:
                    lambdas.append(x[0])
                    metrics.append(metric)
                    seeds.append(x[2])
            input_job_rates = [3600.0 / x for x in lambdas]

            policies = [label for i in range(len(metrics))]
            data["input_job_rate"] += input_job_rates
            data["metric"] += metrics
            data["seed"] += seeds
            data["policy"] += policies
            if len(input_job_rates) > 0 and extrapolate:
                data["input_job_rate"] += [max(input_job_rates) + 0.2]
                data["metric"] += [105.0]
                data["seed"] += [0]
                data["policy"] += [label]

    sns.lineplot(x='input_job_rate', y='metric', style='policy',
                 hue='policy',
                 data=data, ci='sd',
                 markers=True)

    ax.set_xlabel("Input job rate (jobs/hr)")
    ax.set_ylabel(metric_label)
    ax.set_xlim([0, xmax])
    ax.set_ylim([0, ymax])
    sns.despine()
    
    leg = plt.legend(loc='upper left', frameon=False)
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    bb.y0 += 0.1
    bb.y1 += 0.1
    leg.set_bbox_to_anchor(bb, transform=ax.transAxes)

    if output_filename is not None:
        with PdfPages(output_filename) as pdf:
            pdf.savefig(bbox_inches='tight')
    
    plt.show()


def plot_metric_vs_inverse_lambda_different_metric_fns(logfile_paths,
                                                       labels,
                                                       v100s, p100s, k80s,
                                                       policies, metric_fns,
                                                       metric_fn_labels,
                                                       metric_label,
                                                       xmin=0,
                                                       xmax=None,
                                                       ymin=0,
                                                       ymax=None,
                                                       output_filename=None,
                                                       verbose=False):
    from utils import prune

    plt.figure(figsize=(8, 3))
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

    data = {"input_job_rate": [], "metric": [], "seed": [],
            "policy": []}
    print(policies)
    for policy in policies:
        relevant_logfile_paths = list(reversed(prune(
            logfile_paths, v100s, p100s, k80s, policy)))
        for metric_fn_label, metric_fn in zip(metric_fn_labels, metric_fns):
            lambdas = [x[0] for x in relevant_logfile_paths]
            input_job_rates = [3600.0 / x for x in lambdas]
            metrics = [metric_fn(x[1]) for x in relevant_logfile_paths]
            seeds = [x[2] for x in relevant_logfile_paths]
            policies = [labels[policy] + " (%s)" % metric_fn_label
                        for i in range(len(metrics))]

            import pandas as pd
            data["input_job_rate"] += input_job_rates
            data["metric"] += metrics
            data["seed"] += seeds
            data["policy"] += policies
    if verbose:
        import pandas as pd
        df = pd.DataFrame(data)
        print(df.groupby(["policy", "input_job_rate"]).mean())

    sns.lineplot(x='input_job_rate', y='metric', style='policy',
                 hue='policy',
                 data=data, ci='sd',
                 markers=True)

    ax.set_xlabel("Input job rate (jobs/hr)")
    ax.set_ylabel(metric_label)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    sns.despine()
    
    leg = plt.legend(frameon=False)
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    bb.y0 += 0.22
    bb.y1 += 0.22
    leg.set_bbox_to_anchor(bb, transform=ax.transAxes)
    
    if output_filename is not None:
        with PdfPages(output_filename) as pdf:
            pdf.savefig(bbox_inches='tight')
    
    plt.show()


def plot_jct_cdf(logfile_paths,
                 labels,
                 v100s, p100s, k80s,
                 max_input_job_rate,
                 policies,
                 min_job_id, max_job_id,
                 partition=True,
                 finish_time_fairness=False,
                 output_directory=None):
    from utils import get_jcts, prune
    
    lambdas = list(set([x[5] for x in logfile_paths]))
    lambdas.sort(reverse=True)
    print(policies)

    for l in lambdas:
        handles_in_legend = []
        labels_in_legend = []
        
        input_job_rate = 3600.0 / l
        if input_job_rate > max_input_job_rate:
            continue
        print("Input job rate: %.2f" % input_job_rate)
        
        plt.figure(figsize=(8, 3))
        if partition:
            axes = [
                plt.subplot2grid((1, 2), (0, 0), rowspan=1),
                plt.subplot2grid((1, 2), (0, 1), rowspan=1),
            ]
            titles = ["Short jobs", "Long jobs"]
        else:
            axes = [
                plt.subplot2grid((1, 1), (0, 0), rowspan=1),
            ]
            titles = [None]
            if not finish_time_fairness:
                axes.append(axes[0].inset_axes([0.4, 0.2, 0.5, 0.6]))
                titles.append(None)

        if finish_time_fairness:
            relevant_logfile_paths = list(reversed(prune(
                logfile_paths, v100s, p100s, k80s, "isolated", seed=0)))
            relevant_logfile_paths = [x for x in relevant_logfile_paths
                                      if x[0] == l]
            if len(relevant_logfile_paths) != 1:
                continue
            isolated_jcts = get_jcts(relevant_logfile_paths[0][1],
                                     seed=0,
                                     min_job_id=min_job_id,
                                     max_job_id=max_job_id)
            isolated_jcts.sort(key=lambda x: x[1])
        linestyles = ['--', '-.', ':', '--', '-.']
        for i, policy in enumerate(policies):
            relevant_logfile_paths = list(reversed(prune(
                logfile_paths, v100s, p100s, k80s, policy, seed=0)))
            relevant_logfile_paths = [x for x in relevant_logfile_paths
                                      if x[0] == l]
            if len(relevant_logfile_paths) != 1:
                continue
            jcts = get_jcts(relevant_logfile_paths[0][1],
                            seed=0,
                            min_job_id=min_job_id,
                            max_job_id=max_job_id)
            jcts.sort(key=lambda x: x[1])
            partition_point = int(len(jcts) * 0.8)
            if finish_time_fairness:
                jcts = [x[0] / y[0] for (x, y) in zip(jcts, isolated_jcts)]
            else:
                jcts = [x[0] for x in jcts]
                
            print("%s: %.2f" % (policy, np.mean(jcts)))
            if partition:
                jcts = np.split(np.array(jcts), [partition_point])
            else:
                jcts = [np.array(jcts), np.array(jcts)]
            for j, (ax, jcts_segment) in enumerate(zip(axes, jcts)):
                jcts_segment.sort()
                percentiles = [(k+1) / len(jcts_segment)
                               for k in range(len(jcts_segment))]

                if "Gavel" in labels[policy]:
                    handle = ax.plot(jcts_segment, percentiles,
                                     color=current_palette[i],
                                     linestyle='-',
                                     linewidth=3)
                else:
                    handle = ax.plot(jcts_segment, percentiles,
                                     color=current_palette[i],
                                     linestyle=linestyles[i])
                if j == 0:
                    handles_in_legend.append(handle[0])
                    labels_in_legend.append(labels[policy])

        for i, (ax, title) in enumerate(zip(axes, titles)):
            if finish_time_fairness:
                if partition:
                    ax.set_xlabel("FTF" + "\n" + title)
                else:
                    ax.set_xlabel("FTF")
                ax.set_xlim([0, 4])
                ax.set_xticks([0, 1, 2, 3, 4])
                ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            else:
                if partition:
                    ax.set_xlabel("JCT (hrs)" + "\n" + title)
                else:
                    if i == 0:
                        ax.set_xlabel("JCT (hrs)")
                if not partition:
                    if i == 0:
                        ax.set_xlim([0, 500])
                        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    else:
                        ax.set_xlim([0, 25])
                        ax.set_xticks([0, 5, 10, 15, 20, 25])
                        ax.set_yticks([0, 0.33, 0.67, 1.0])
            if i == 0:
                ax.set_ylabel("Fraction of jobs")
            if partition:
                if i > 0:
                    ax.set_yticklabels(["", "", "", "", "", ""])
            ax.set_ylim([0, 1.0])
        sns.despine()
        
        if not partition and not finish_time_fairness:
            axes[0].indicate_inset_zoom(axes[1], linewidth=3)
    
        leg = plt.figlegend(handles=handles_in_legend,
                            labels=labels_in_legend,
                            ncol=3, frameon=False,
                            loc='upper center')

        bb = leg.get_bbox_to_anchor().inverse_transformed(
            axes[0].transAxes)
        bb.y0 += 0.22
        bb.y1 += 0.22
        leg.set_bbox_to_anchor(bb, transform=axes[0].transAxes)
        
        if output_directory is not None:
            output_filename = os.path.join(output_directory,
                                           "input_job_rate=%d.pdf" % (input_job_rate * 10))
            with PdfPages(output_filename) as pdf:
                pdf.savefig(bbox_inches='tight')
        
        plt.show()
