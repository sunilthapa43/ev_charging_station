keep_features = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp',
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
            'Fwd Header Length', 'Bwd Header Length', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'Label']


DoS_Types = ['DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye']
Brute_Force_Types = ['FTP-Patator', 'SSH-Patator']
Web_Attack_types = ['Web Attack \x96 Brute Force', 'Web Attack \x96 XSS', 'Web Attack \x96 Sql Injection']
Others =['Heartbleed', 'Infiltration', 'Bot']