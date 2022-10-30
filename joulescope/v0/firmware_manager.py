# Copyright 2019 Jetperch LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from joulescope.v0.bootloader import Bootloader
from joulescope.v0.driver import bootloader_go
from zipfile import ZipFile
import monocypher
import binascii
import json
import logging
import threading


log = logging.getLogger(__name__)


SIGNING_KEY_PUBLIC = binascii.unhexlify(b'32fe2bed04bbc42fe1b382e0371ba95ec2947045e8d919e49fdef601e24c105e')


VERSIONS = {
    'namespace': 'joulescope',
    'type': 'firmware-versions',
    'version': 1,
    'data': {
        'format': 'js110_{version}.img',
        # alpha
        # beta
        'production': '1.3.2',
        'available': ['1.3.2']
    }
}


def load(path):
    with ZipFile(path, mode='r') as f_zip:
        with f_zip.open('index.json', 'r') as f:
            index_bytes = f.read()
        with f_zip.open('index.sig', 'r') as f:
            index_sig = binascii.unhexlify(f.read())

        if not monocypher.signature_check(index_sig, SIGNING_KEY_PUBLIC, index_bytes):
            log.warning('integrity check failed: index.json')
            return None

        index = json.loads(index_bytes.decode('utf-8'))
        for image in index['target']['images']:
            with f_zip.open(index['data'][image]['image'], 'r') as f:
                index['data'][image]['image'] = f.read()
            sig = binascii.unhexlify(index['data'][image]['signature'])
            if not monocypher.signature_check(sig, SIGNING_KEY_PUBLIC, index['data'][image]['image']):
                log.warning('integrity check failed: %s' % (image, ))
                return None
    return index


def version_required(release=None):
    release = 'production' if release is None else str(release)
    v = VERSIONS['data'][release]
    return tuple([int(x) for x in v.split('.')])


class UpgradeThread:

    def __init__(self, device, image, progress_cbk, stage_cbk, done_cbk):
        self.device = device
        self.image = image
        self.progress_cbk = progress_cbk
        self.stage_cbk = stage_cbk
        self.done_cbk = done_cbk

    def run(self):
        d = None
        try:
            d = upgrade(self.device, self.image, self.progress_cbk, self.stage_cbk)
        finally:
            self.done_cbk(d)


def upgrade(device, image, progress_cbk=None, stage_cbk=None, done_cbk=None):
    """Full upgrade the device's firmware.

    :param device: The :class:`Device` or class:`bootloader.Bootloader` instance
        that must already be open.
    :param image: The image returned by :func:`load`.  Alternatively, a path
        suitable for :func:`load`.
    :param progress_cbk:  The optional Callable[float] which is called
        with the progress fraction from 0.0 to 1.0
    :param stage_cbk: The optional Callable[str] which is called with a
        meaningful stage description for each stage of the upgrade process.
    :param done_cbk: The optional Callback[object] which is called with
        the device on success or None on failure.  If done_cbk is provided,
        then run the upgrade in its own thread.
    :return: The :class:`Device` which is closed.
    raise IOError: on failure.
    """

    if done_cbk is not None:
        t = UpgradeThread(device, image, progress_cbk, stage_cbk, done_cbk)
        thread = threading.Thread(name='fw_upgrade', target=t.run)
        thread.start()
        return thread

    try:
        cbk_data = {
            'stages': [
                ('Load image', 0.05),
                ('Start bootloader', 0.05),
                ('Program application', 0.1),
                ('Start application', 0.05),
                ('Program sensor', 0.75),
                ('Done', 0.0),
            ],
            'stage': -1,
        }

        def next_stage():
            cbk(1.0)
            cbk_data['stage'] += 1
            s, _ = cbk_data['stages'][cbk_data['stage']]
            log.info('firmware_upgrade: %s', s)
            if stage_cbk:
                stage_cbk(s)

        def cbk(progress):
            previous = 0.0
            for idx in range(cbk_data['stage']):
                previous += cbk_data['stages'][idx][1]
            current = cbk_data['stages'][cbk_data['stage']][1]
            if progress_cbk:
                progress_cbk(previous + progress * current)

        next_stage()
        if isinstance(image, str):
            image = load(image)

        next_stage()
        if not isinstance(device, Bootloader):
            b, _ = device.bootloader(progress_cbk=cbk)
        else:
            b = device
        try:
            next_stage()
            rc = b.firmware_program(image['data']['controller']['image'], progress_cbk=cbk)
            if rc:
                raise IOError('controller firmware programming failed: %d', rc)
            next_stage()
        except Exception:
            b.close()
            raise
        d = bootloader_go(b, progress_cbk=cbk)
        next_stage()
        d.open()
        try:
            d.sensor_firmware_program(image['data']['sensor']['image'], progress_cbk=cbk)
        finally:
            d.close()
        if done_cbk:
            done_cbk(d)
        return d
    except Exception:
        if done_cbk:
            done_cbk(None)
        raise


def run():
    import sys
    from joulescope.v0.driver import scan_require_one
    with scan_require_one() as d:
        upgrade(d, sys.argv[1], progress_cbk=print, stage_cbk=print)


if __name__ == '__main__':
    run()
