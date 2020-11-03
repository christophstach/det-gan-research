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

    checkpoint_group = parser.add_argument_group('checkpoint selection')
    checkpoint_type_group = checkpoint_group.add_mutually_exclusive_group(required=True)
    checkpoint_type_group.add_argument(
        '-e', '--experiment-id', type=int, help='Defines the experiment to download models from.'
    )
    checkpoint_type_group.add_argument(
        '-t', '--trial-id', type=int, help='Defines the trial to download models from.'
    )
    checkpoint_type_group.add_argument(
        '-c', '--checkpoint-uuid', type=str, help='The checkpoint uuid to download.'
    )

    checkpoint_group.add_argument(
        '--smaller-is-better', action='store_true',
        help='Prefer checkpoints with smaller searcher metric. '
             'This is only evaluated, if --sort-by is given. (default: False)'
    )
    checkpoint_group.add_argument(
        '--sort-by', metavar='SORTBY', type=str, default=None,
        help='The metric to sort the checkpoints with. Defaults to the metric specified in the experiment.'
    )
    checkpoint_group.add_argument(
        '-n', '--num-checkpoints', metavar='NUMCHECKPOINTS', type=int, default=1,
        help='Number of checkpoints to download (default: 1)'
    )

    execution_group = parser.add_argument_group('execution')
    execution_group.add_argument('--dry', action='store_true', help='Just output actions instead of executing them.')
    execution_group.add_argument('--silent', action='store_true', help='Set this to prevent debug information.')
    execution_group.add_argument(
        '-d', '--local-dir', type=str, default=LOCAL_CHECKPOINT_FOLDER,
        help='Local path where to store the downloaded checkpoint data (default: {}).'.format(LOCAL_CHECKPOINT_FOLDER)
    )

    cluster_group = parser.add_argument_group('cluster')
    cluster_group.add_argument(
        '-m', '--master-url', metavar='MASTERURL', type=str, default='dt1.f4.htw-berlin.de',
        help='The url of the determined master (default: dt1.f4.htw-berlin.de).'
    )
    cluster_group.add_argument(
        '--determined-user', metavar='DETERMINEDUSER', type=str, default=None,
        help='Your determined user. If not given this script tries to identify your user automatically. '
             'You can change this with det login <username>.'
    )
    cluster_group.add_argument(
        '--storage-server', metavar='STORAGESERVER', type=str, default='avocado01.f4.htw-berlin.de',
        help='The storage server for the scp command (default: avocado01.f4.htw-berlin.de).'
    )
    cluster_group.add_argument(
        '-u', '--storage-user', metavar='STORAGEUSER', type=str, required=True,
        help='The user to use to connect to the storage server (required).'
    )
    cluster_group.add_argument(
        '-i', '--identity-file', metavar='IDENTITYFILE', type=str, default=None,
        help='SSH Identity File to use for scp authentication.'
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

    determined = Determined(master='https://{}:8443'.format(args.master_url), user=args.determined_user)
    if args.experiment_id is not None:
        experiment = determined.get_experiment(args.experiment_id)
        checkpoints = experiment.top_n_checkpoints(
            args.num_checkpoints,
            sort_by=args.sort_by,
            smaller_is_better=args.smaller_is_better
        )
    elif args.trial_id is not None:
        if args.num_checkpoints != 1:
            raise ValueError('Argument "--num-checkpoints" is invalid with trial id.')
        trial = determined.get_trial(args.trial_id)
        checkpoints = [trial.top_checkpoint(sort_by=args.sort_by, smaller_is_better=args.smaller_is_better)]
    elif args.checkpoint_uuid is not None:
        if args.num_checkpoints != 1:
            raise ValueError('Argument "--num-checkpoints" is invalid with checkpoint uuid')
        if args.sort_by is not None:
            raise ValueError('Argument "--sort-by" is invalid with checkpoint uuid')
        if args.smaller_is_better:
            raise ValueError('Argument "--smaller-is-better" is invalid with checkpoint uuid')
        checkpoints = [determined.get_checkpoint(args.checkpoint_uuid)]
    else:
        raise ValueError('Either experiment_id or trial_id should be given')

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
