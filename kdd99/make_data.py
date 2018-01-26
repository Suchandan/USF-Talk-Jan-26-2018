import argparse

from helper_functions import kdd99_test_train_split


# ==============================================================================
# Main
# ==============================================================================

def main(FLAGS):
    kdd99_test_train_split(nrows=FLAGS.num_rows)


# ==============================================================================
# __name__ == __main__
# ==============================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_rows',
        type=int,
        default=None,
        help='Number of KDD99 rows to use for generating the data. Default: use all rows.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
