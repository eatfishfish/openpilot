import json
import os

import pytest
from Crypto.PublicKey import RSA
from pathlib import Path

from openpilot.common.params import Params
from openpilot.system.athena.registration import register, UNREGISTERED_DONGLE_ID
from openpilot.system.athena.tests.helpers import MockResponse
from openpilot.system.hardware.hw import Paths


class TestRegistration:

  def setup_method(self):
    # clear params and setup key paths
    self.params = Params()

    persist_dir = Path(Paths.persist_root()) / "comma"
    persist_dir.mkdir(parents=True, exist_ok=True)

    self.priv_key = persist_dir / "id_rsa"
    self.pub_key = persist_dir / "id_rsa.pub"
    self.dongle_id = persist_dir / "dongle_id"

  def _generate_keys(self):
    self.pub_key.touch()
    k = RSA.generate(2048)
    with open(self.priv_key, "wb") as f:
      f.write(k.export_key())
    with open(self.pub_key, "wb") as f:
      f.write(k.publickey().export_key())

  def test_valid_cache(self, mocker):
    # if all params are written, return the cached dongle id.
    # should work with a dongle ID on either /persist/ or normal params
    self._generate_keys()

    dongle = "DONGLE_ID_123"
    m = mocker.patch("openpilot.system.athena.registration.api_get", autospec=True)
    for persist, params in [(True, True), (True, False), (False, True)]:
      self.params.put("DongleId", dongle if params else "")
      with open(self.dongle_id, "w") as f:
        f.write(dongle if persist else "")
      assert register() == dongle
      assert not m.called

  def test_no_keys(self, mocker):
    # missing keys should still register without key auth
    m = mocker.patch("openpilot.system.athena.registration.api_get", autospec=True)
    dongle = "DONGLE_ID_123"
    m.return_value = MockResponse(json.dumps({'dongle_id': dongle}), 200)
    mocker.patch("openpilot.system.athena.registration.HARDWARE.get_serial", return_value="serial")
    mocker.patch("openpilot.system.athena.registration.HARDWARE.get_imei", return_value="imei")
    dongle = register()
    assert m.call_count == 1
    assert self.params.get("DongleId") == dongle
    assert "public_key" not in m.call_args.kwargs
    assert "register_token" not in m.call_args.kwargs

  def test_missing_cache(self, mocker):
    # keys exist but no dongle id
    self._generate_keys()
    m = mocker.patch("openpilot.system.athena.registration.api_get", autospec=True)
    dongle = "DONGLE_ID_123"
    m.return_value = MockResponse(json.dumps({'dongle_id': dongle}), 200)
    assert register() == dongle
    assert m.call_count == 1

    # call again, shouldn't hit the API this time
    assert register() == dongle
    assert m.call_count == 1
    assert self.params.get("DongleId") == dongle

  def test_unregistered(self, mocker):
    # keys exist, but unregistered
    self._generate_keys()
    m = mocker.patch("openpilot.system.athena.registration.api_get", autospec=True)
    m.return_value = MockResponse(None, 402)
    dongle = register()
    assert m.call_count == 1
    assert dongle == UNREGISTERED_DONGLE_ID
    assert self.params.get("DongleId") == dongle

  @pytest.mark.nocapture
  def test_real_registration(self):
    if os.getenv("REAL_REGISTRATION") != "1":
      pytest.skip("set REAL_REGISTRATION=1 to call the real registration API")

    self.params.remove("DongleId")
    if self.dongle_id.exists():
      self.dongle_id.unlink()

    dongle = register()
    assert dongle is not None
    assert self.params.get("DongleId") == dongle
