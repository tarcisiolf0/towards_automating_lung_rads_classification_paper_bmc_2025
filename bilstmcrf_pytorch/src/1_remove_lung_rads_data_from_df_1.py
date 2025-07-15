import pandas as pd

index_to_remove = [540, 541, 546, 550, 552, 554, 556, 558, 559, 577, 583, 589, 591, 595, 598, 613, 615, 616, 618, 627, 629, 637, 640, 646, 650, 659, 660, 662, 665, 666, 673, 677, 684, 691, 693, 694, 697, 699, 701, 702, 703, 706, 707, 712, 713, 719, 720, 725, 726, 727, 731, 734, 741, 744, 747, 749, 751, 752, 753, 754, 759, 760, 771, 774, 776, 792, 795, 797, 806, 811, 813, 818, 819, 820, 821, 822, 830, 832, 834, 836, 839, 847, 848, 851, 853, 864, 865, 867, 871, 873, 874, 875, 877, 880, 881, 885, 886, 893, 896, 900]
input_file_path = "bilstmcrf_pytorch/data/df_tokens_labeled_iob.csv"
output_file_path = "bilstmcrf_pytorch/data/df_tokens_labeled_iob_with_lr_idx.csv"

# Load the DataFrame
df = pd.read_csv(input_file_path, encoding='utf-8')

# Remove the rows with the specified indices
#filterd_df = df[~df['report_index'].isin(index_to_remove)]
filterd_df = df[df['report_index'].isin(index_to_remove)]

filterd_df.to_csv(output_file_path, index=False, encoding='utf-8')