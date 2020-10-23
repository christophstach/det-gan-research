#!/usr/bin/env python


import argparse
import json
import os
import subprocess

from determined.experimental import Determined


LOCAL_CHECKPOINT_FOLDER = './checkpoints'
SCP_TIMEOUT = 5


def parse_args():
    parser = argparse.ArgumentParser(description='Download determined models')

    required_group = parser.add_argument_group('required')
    required_group.add_argument(
        'experiment_id', metavar='experiment-id', type=int, help='The experiment to download models from.'
    )
    required_group.add_argument(
        '--storage-user', metavar='STORAGEUSER', type=str, required=True,
        help='The user to use to connect to the storage server (required).'
    )

    execution_group = parser.add_argument_group('execution')
    execution_group.add_argument('--dry', action='store_true', help='Just output actions instead of executing them.')
    execution_group.add_argument('--silent', action='store_true', help='Set this to prevent debug information.')
    execution_group.add_argument(
        '--local-dir', type=str, default=LOCAL_CHECKPOINT_FOLDER,
        help='Local path where to store the downloaded checkpoint data (default: {})'.format(LOCAL_CHECKPOINT_FOLDER)
    )

    checkpoint_group = parser.add_argument_group('checkpoint selection')
    checkpoint_group.add_argument(
        '--lower-is-better', action='store_true', help='Prefer checkpoints with lower searcher metric.'
    )
    checkpoint_group.add_argument(
        '--num-checkpoints', metavar='NUMCHECKPOINTS', type=int, default=1,
        help='Number of checkpoints to download (default: 1)'
    )

    cluster_group = parser.add_argument_group('cluster')
    cluster_group.add_argument(
        '--master-url', metavar='MASTERURL', type=str, default='dt1.f4.htw-berlin.de',
        help='The url of the determined master (default: dt1.f4.htw-berlin.de).'
    )
    cluster_group.add_argument(
        '--identity-file', metavar='IDENTITYFILE', type=str, default=None,
        help='Identity File to use for scp authentication.'
    )
    cluster_group.add_argument(
        '--storage-server', metavar='STORAGESERVER', type=str, default='avocado01.f4.htw-berlin.de',
        help='The storage server for the scp command (default: avocado01.f4.htw-berlin.de).'
    )

    return parser.parse_args()


def prepare_checkpoint_folder(dry, silent, local_dir):
    if not os.path.exists(local_dir):
        if dry or not silent:
            print('INFO: mkdir {}'.format(local_dir))
        if not dry:
            os.mkdir(local_dir)


def main():
    args = parse_args()

    prepare_checkpoint_folder(args.dry, args.silent, args.local_dir)
    experiment = Determined(master='https://dt1.f4.htw-berlin.de:8443').get_experiment(args.experiment_id)
    checkpoints = experiment.top_n_checkpoints(2)

    errors = []

    for index, checkpoint in enumerate(checkpoints):
        local_path = os.path.join(args.local_dir, checkpoint.uuid)
        if os.path.exists(local_path):
            if not args.silent:
                print('INFO: directory "{}" already exists. Skipping.'.format(local_path))
        else:
            if not args.silent:
                print('\nINFO: running scp for checkpoint {} of {}'.format(index+1, len(checkpoints)))
            scp_command = ['scp', '-r', '-o', 'ConnectTimeout={}'.format(SCP_TIMEOUT)]
            if args.identity_file is not None:
                identity_file = os.path.expanduser(args.identity_file)
                scp_command.extend(['-i', identity_file])
            scp_command.extend([
                '{}@{}:/data/determined/shared_fs/checkpoints/{}'
                .format(args.storage_user, args.storage_server, checkpoint.uuid),
                local_path
            ])

            if args.dry or not args.silent:
                print('executing ' + ' '.join(scp_command))
            if not args.dry:
                stdout_arg = None if not args.silent else subprocess.DEVNULL
                scp_result = subprocess.run(scp_command, stdout=stdout_arg, stderr=stdout_arg, stdin=stdout_arg)
                if scp_result.returncode != 0:
                    print('ERROR: Failed to execute scp. Are you connected to VPN?')
                    errors.append('Could not download checkpoint "{}"'.format(checkpoint.uuid))
                    continue

            info_file_path = os.path.join(local_path, 'metadata.json')
            if args.dry or not args.silent:
                print('INFO: Dumping info file "{}"'.format(info_file_path))
            if not args.dry:
                with open(info_file_path, 'w') as info_file:
                    meta_info = {
                        'determined_version': checkpoint.determined_version,
                        'framework': checkpoint.framework,
                        'format': checkpoint.format,
                        'experiment_id': checkpoint.experiment_id,
                        'trial_id': checkpoint.trial_id,
                        'hparams': checkpoint.hparams,
                        'experiment_config': checkpoint.experiment_config,
                        'metadata': checkpoint.metadata,
                    }
                    json.dump(meta_info, info_file, sort_keys=True, indent=4)

    if not args.silent:
        if errors:
            print('\nINFO: not all models could be downloaded:\n{}'.format('\n'.join(errors)))
        else:
            print('\nINFO: all models downloaded under directory "{}"'.format(args.local_dir))


if __name__ == '__main__':
    main()
