from quaisr import Block
from quaisr.types import (
    ANY_TYPE,
    BOOLEAN,
    QuaisrDictionaryType,
    QuaisrFileType,
    QuaisrListType,
    FLOAT,
    INTEGER,
    STRING,
    QuaisrRecordType,
    QuaisrOptionalType
)

from nearest_neighbour_gaussian_process.core import train_nngp

if __name__ == "__main__":
    # Test the block type signature with "python main.py --generate"
    block = Block()

    optimiser_record_type = QuaisrRecordType({
        "k" : INTEGER,
        "training_batch_size" : INTEGER,
        "training_epochs" : INTEGER,
        "learning_rate" : FLOAT,
        "milestones": QuaisrListType(INTEGER),
        # Number of inducing points (mini-batch k-means centroids). If >= number
        # of data points, every data point is used as an inducing point.
        "ninducing": INTEGER,
        })

    kernel_record_type = QuaisrRecordType({
        "kernel" : STRING,
        "options" : QuaisrDictionaryType(STRING, ANY_TYPE),
    })

    # Options for the automatic kernel, used only when "kernels" is omitted.
    # Omit the record to train the stationary kernel every earlier run trained.
    kernel_options_type = QuaisrRecordType({
        # Cells across the survey for the amplitude field on the short-scale
        # component. Omit or 0 to keep that component stationary.
        "modulation_control_count": QuaisrOptionalType(INTEGER),
        # Symmetric bound on the log weight of each control point. Omit for
        # 1.0. The bumps overlap, so the amplitude contrast across the survey
        # comes out wider than the bound itself.
        "modulation_log_amplitude": QuaisrOptionalType(FLOAT),
    })

    likelihood_record_type = QuaisrRecordType({
        "name" : STRING,
        "noise" : FLOAT,
        # NaN recovery: on a NaN/non-PSD run the fixed noise is multiplied by
        # noise_growth and training is retried, up to max_noise_retries times.
        "noise_growth" : QuaisrOptionalType(FLOAT),
        "max_noise_retries" : QuaisrOptionalType(INTEGER),
    })

    vecchia_options_type = QuaisrRecordType({
        "use_hierarchical_vecchia": BOOLEAN,
        "n_vecchia_levels": INTEGER,
        "k_cross_ratio": FLOAT,
        "k_l0_fraction": FLOAT,
    })

    # Point caps for the post-training metrics and plots. Omit either field to
    # keep the 100,000-point default; set it to 0 or less to use every point.
    # validation_batch_size caps points per posterior batch; peak memory there is
    # O(batch * k^2). Omit it to derive one from k and free GPU memory.
    validation_options_type = QuaisrRecordType({
        "max_validation_points": QuaisrOptionalType(INTEGER),
        "max_diagnostics_points": QuaisrOptionalType(INTEGER),
        "validation_batch_size": QuaisrOptionalType(INTEGER),
        # Cells across for the predictive nugget field. Omit or 0 to report the
        # single survey-wide nugget, as every earlier run did.
        "nugget_control_count": QuaisrOptionalType(INTEGER),
    })

    block.add_input("training_samples_file", QuaisrFileType("application/octet-stream"))
    block.add_input("test_samples_file", QuaisrFileType("application/octet-stream"))
    block.add_input("kernels", QuaisrOptionalType(QuaisrListType(kernel_record_type)))
    block.add_input("kernel_options", QuaisrOptionalType(kernel_options_type))
    block.add_input("likelihood", likelihood_record_type)
    block.add_input("optimiser", optimiser_record_type)
    block.add_input("vecchia_options", QuaisrOptionalType(vecchia_options_type))
    block.add_input("validation_options", QuaisrOptionalType(validation_options_type))

    block.add_output("loss_stream", QuaisrListType(FLOAT), stream=True)
    block.add_output("model_file", QuaisrFileType("application/octet-stream"))
    block.add_output("validation_metrics", QuaisrDictionaryType(STRING, FLOAT))
    block.add_output("validation_plots", QuaisrDictionaryType(STRING, QuaisrFileType("text/html")))

    block.add_output("diagnostics_metrics", QuaisrDictionaryType(STRING, FLOAT))
    block.add_output("diagnostics_plots", QuaisrDictionaryType(STRING, QuaisrFileType("text/html")))
    block.add_output("time_left", QuaisrListType(INTEGER), stream=True)
    block.add_output("log", STRING, stream=True)

    block.set_entrypoint(train_nngp).start()
