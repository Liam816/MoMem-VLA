"""CPU tests for the stage grid and real SimpleMemVLA Trainer save/resume flow.

Run in the project's existing environment, never on CUDA:
  CUDA_VISIBLE_DEVICES='' python test_checkpoint_schedule.py \
    --repo /path/to/SimpleMemVLA --output /path/to/new/test-output
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--repo")
parser.add_argument("--output")
parser.add_argument("--unit-only", action="store_true")
OPTIONS, UNITTEST_ARGS = parser.parse_known_args()
if OPTIONS.repo is None:
    # Also support ordinary unittest discovery from a canonical repo/tests path.
    inferred = Path(__file__).resolve().parents[1]
    if (inferred / "configs/sft_params.py").is_file() and (inferred / "simplememvla").is_dir():
        OPTIONS.repo = str(inferred)
if OPTIONS.repo:
    sys.path.insert(0, OPTIONS.repo)
    from simplememvla.compat import apply_fla_torch_compat
    apply_fla_torch_compat()
    from simplememvla.training.checkpoint_schedule import (
        UniformCheckpointCallback,
        build_checkpoint_steps,
        validate_checkpoint_schedule,
    )
else:
    from checkpoint_schedule import (
        UniformCheckpointCallback,
        build_checkpoint_steps,
        validate_checkpoint_schedule,
    )

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_callback import DefaultFlowCallback
from transformers.trainer_utils import SaveStrategy

REPORT = {"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "integration": {}}
OUTPUT = Path(OPTIONS.output or tempfile.mkdtemp(prefix="uniform-checkpoint-tests-"))
OUTPUT.mkdir(parents=True, exist_ok=True)


def callback_args(path, fraction=0.1, count=5, strategy="steps", main=True):
    return SimpleNamespace(
        output_dir=str(path), save_first_fraction=fraction,
        save_num_checkpoints=count, save_strategy=strategy,
        load_best_model_at_end=False, process_index=0 if main else 1,
    )


class GridTests(unittest.TestCase):
    def test_50000_golden(self):
        self.assertEqual(build_checkpoint_steps(50000, 0.1, 5), [5000, 16250, 27500, 38750, 50000])

    def test_decimal_ceil_and_half_up(self):
        self.assertEqual(build_checkpoint_steps(100, 0.07, 4), [7, 38, 69, 100])
        self.assertEqual(build_checkpoint_steps(20, Decimal("0.1"), 5), [2, 7, 11, 16, 20])
        self.assertEqual(build_checkpoint_steps(99, 0.1, 5), [10, 32, 55, 77, 99])
        self.assertEqual(build_checkpoint_steps(50001, 0.1, 5), [5001, 16251, 27501, 38751, 50001])

    def test_dense_and_single_stage(self):
        self.assertEqual(build_checkpoint_steps(5, 0.1, 5), [1, 2, 3, 4, 5])
        self.assertEqual(build_checkpoint_steps(20, 1, 1), [20])
        self.assertEqual(build_checkpoint_steps(1, 0.1, 1), [1])
        self.assertEqual(build_checkpoint_steps(3, 0.99, 1), [3])

    def test_disabled_and_invalid_pairs(self):
        self.assertFalse(validate_checkpoint_schedule(None, None))
        self.assertEqual(build_checkpoint_steps(20, None, None), [])
        for fraction, count in [(None, 5), (0.1, None), (0, 5), (-0.1, 5), (1.1, 5),
                                (float("nan"), 5), (float("inf"), 5), (True, 5),
                                ("0.1", 5), (0.1, False), (0.1, 0), (0.1, -1), (0.1, 5.0)]:
            with self.subTest(fraction=fraction, count=count), self.assertRaises(ValueError):
                validate_checkpoint_schedule(fraction, count)
        for total, fraction, count in [(0, 0.1, 5), (True, 0.1, 5), (5.0, 0.1, 5),
                                       (20, 0.1, 1), (5, 0.1, 6), (20, 1, 2)]:
            with self.subTest(total=total, fraction=fraction, count=count), self.assertRaises(ValueError):
                build_checkpoint_steps(total, fraction, count)


class CallbackTests(unittest.TestCase):
    def test_override_default_step_epoch_final_and_deduplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            args=callback_args(directory)
            state=TrainerState(max_steps=20, is_world_process_zero=True)
            control=TrainerControl()
            cb=UniformCheckpointCallback()
            cb.on_train_begin(args,state,control)
            for step, expected in [(1,False),(2,True),(3,False),(7,True),(20,True)]:
                state.global_step=step;control.should_save=True
                cb.on_step_end(args,state,control)
                self.assertEqual(control.should_save,expected)
                if expected:
                    cb.on_save(args,state,control)
                    cb.on_step_end(args,state,control)
                    self.assertFalse(control.should_save)
                control.should_save=True
                cb.on_epoch_end(args,state,control)
                self.assertFalse(control.should_save)

    def test_disabled_passthrough_and_rank_zero_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            args=callback_args(directory,None,None)
            state=TrainerState(max_steps=20,is_world_process_zero=True)
            control=TrainerControl(should_save=True)
            cb=UniformCheckpointCallback();cb.on_train_begin(args,state,control)
            cb.on_step_end(args,state,control);self.assertTrue(control.should_save)
            cb.on_epoch_end(args,state,control);self.assertTrue(control.should_save)
            self.assertFalse((Path(directory)/"checkpoint_schedule.json").exists())
            args=callback_args(directory,main=False)
            state.is_world_process_zero=False
            cb.on_train_begin(args,state,control)
            self.assertFalse((Path(directory)/"checkpoint_schedule.json").exists())

    def test_resume_plan_and_budget_change_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=callback_args(root)
            state=TrainerState(max_steps=20,global_step=0,is_world_process_zero=True)
            control=TrainerControl()
            cb=UniformCheckpointCallback();cb.on_train_begin(args,state,control)
            (root/"checkpoint-2").mkdir();(root/"checkpoint-7").mkdir()
            state.global_step=7
            cb=UniformCheckpointCallback();cb.on_train_begin(args,state,control)
            plan=json.loads((root/"checkpoint_schedule.json").read_text())
            self.assertEqual(plan["future_steps"],[11,16,20])
            self.assertEqual(plan["skipped_past_or_current_steps"],[2,7])
            self.assertEqual(plan["existing_past_checkpoints"],[2,7])
            self.assertFalse(list(root.glob("checkpoint_schedule_history/*")))
            cb.on_step_end(args,state,control);self.assertFalse(control.should_save)
            state.max_steps=40
            cb.on_train_begin(args,state,control)
            archived=list(root.glob("checkpoint_schedule_history/*.json"))
            self.assertEqual(len(archived),1)
            self.assertEqual(json.loads(archived[0].read_text())["total_optimizer_steps"],20)

    def test_incompatible_best_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            args=callback_args(directory,strategy="best")
            with self.assertRaises(ValueError):
                UniformCheckpointCallback().on_train_begin(args,TrainerState(max_steps=20),TrainerControl())
            args.save_strategy="steps";args.load_best_model_at_end=True
            with self.assertRaises(ValueError):
                UniformCheckpointCallback().on_train_begin(args,TrainerState(max_steps=20),TrainerControl())


class SaveRecorder(TrainerCallback):
    def __init__(self,stop_at=None):
        self.stop_at=stop_at;self.steps=[];self.total_steps=None;self.epoch_ends=[]

    def on_train_begin(self,args,state,control,**kwargs):
        self.total_steps=state.max_steps

    def on_step_end(self,args,state,control,**kwargs):
        if self.stop_at is not None and state.global_step>=self.stop_at:
            control.should_training_stop=True
        return control

    def on_save(self,args,state,control,**kwargs):
        self.steps.append(state.global_step)

    def on_epoch_end(self,args,state,control,**kwargs):
        self.epoch_ends.append(state.global_step)


@unittest.skipIf(OPTIONS.unit_only,"--unit-only requested")
class TrainerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from configs.sft_params import SFTTrainingArguments
        from simplememvla.training.sft_trainer import SimpleMemVLATrainer
        cls.torch=torch;cls.Args=SFTTrainingArguments;cls.Trainer=SimpleMemVLATrainer

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone=torch.nn.Linear(1,1)
                self.config=SimpleNamespace(use_memory=False,keys_to_ignore_at_inference=[])

            def forward(self,input_values,labels):
                predictions=self.backbone(input_values)
                loss=(predictions-labels).square().mean()
                return SimpleNamespace(loss=loss,action_loss=loss,vl_loss=None,mem_norm=None)

        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self,size):self.size=size
            def __len__(self):return self.size
            def __getitem__(self,index):
                value=torch.tensor([index%7/7],dtype=torch.float32)
                return {"input_values":value,"labels":value*0.7+0.2}

        cls.Model=TinyModel;cls.Dataset=TinyDataset

    def args(self,path,**overrides):
        values=dict(output_dir=str(path),max_steps=20,num_train_epochs=3,
                    per_device_train_batch_size=2,gradient_accumulation_steps=2,
                    use_cpu=True,bf16=False,fp16=False,tf32=False,
                    gradient_checkpointing=False,dataloader_num_workers=0,
                    dataloader_persistent_workers=False,report_to=[],
                    logging_strategy="no",disable_tqdm=True,save_strategy="steps",
                    save_steps=1,save_total_limit=1,
                    eval_strategy="no",load_best_model_at_end=False,
                    seed=17,data_seed=17,warmup_steps=2,
                    backbone_lr=0.01,action_head_lr=0.01,memory_lr=0.01,
                    save_first_fraction=0.1,save_num_checkpoints=5,resume=False)
        values.update(overrides)
        return self.Args(**values)

    @staticmethod
    def checkpoints(path):
        return sorted(int(p.name.split('-')[-1]) for p in Path(path).glob('checkpoint-*') if p.is_dir())

    def test_iter_schedule_and_normal_save_payload(self):
        path=OUTPUT/'iter'
        self.assertFalse(path.exists(),f"Refusing existing test output: {path}")
        args=self.args(path)
        self.assertGreaterEqual(args.save_total_limit,5)
        recorder=SaveRecorder()
        trainer=self.Trainer(model=self.Model(),args=args,train_dataset=self.Dataset(13),callbacks=[recorder])
        self.assertTrue(any(isinstance(c,UniformCheckpointCallback) for c in trainer.callback_handler.callbacks))
        trainer.train()
        expected=[2,7,11,16,20]
        self.assertEqual(recorder.steps,expected)
        self.assertEqual(self.checkpoints(path),expected)
        for step in expected:
            folder=path/f'checkpoint-{step}'
            self.assertTrue((folder/'trainer_state.json').exists())
            self.assertTrue((folder/'optimizer.pt').exists())
            self.assertTrue((folder/'scheduler.pt').exists())
            self.assertTrue((folder/'rng_state.pth').exists())
            self.assertTrue((folder/'model.safetensors').exists())
            self.assertEqual(json.loads((folder/'trainer_state.json').read_text())['global_step'],step)
        REPORT['integration']['iter']={'max_steps':trainer.state.max_steps,'saved_steps':recorder.steps,'checkpoint_directories':self.checkpoints(path)}

    def test_epoch_fractional_nondivisible_accumulation(self):
        path=OUTPUT/'epoch'
        self.assertFalse(path.exists(),f"Refusing existing test output: {path}")
        args=self.args(path,max_steps=-1,num_train_epochs=2.5,gradient_accumulation_steps=4,
                       save_first_fraction=0.2,save_num_checkpoints=3,save_total_limit=3)
        # Exercise DefaultFlow's epoch saving too. The custom callback must
        # suppress extra epoch saves, independently of any argument normalization.
        args.save_strategy=SaveStrategy.EPOCH
        recorder=SaveRecorder()
        trainer=self.Trainer(model=self.Model(),args=args,train_dataset=self.Dataset(11),callbacks=[recorder])
        trainer.train()
        plan=json.loads((path/'checkpoint_schedule.json').read_text())
        self.assertEqual(plan['total_optimizer_steps'],recorder.total_steps)
        self.assertEqual(plan['total_optimizer_steps'],trainer.state.max_steps)
        self.assertGreater(recorder.total_steps,0)
        self.assertEqual(len(recorder.steps),3)
        self.assertEqual(len(set(recorder.steps)),3)
        self.assertEqual(recorder.steps[-1],recorder.total_steps)
        self.assertEqual(recorder.steps,plan['checkpoint_steps'])
        self.assertEqual(self.checkpoints(path),recorder.steps)
        self.assertGreater(len(recorder.epoch_ends),1)
        REPORT['integration']['epoch']={'configured_max_steps':-1,'num_train_epochs':2.5,
            'dataset_size':11,'batch_size':2,'gradient_accumulation_steps':4,
            'resolved_total_optimizer_steps':recorder.total_steps,'epoch_end_steps':recorder.epoch_ends,
            'saved_steps':recorder.steps,'checkpoint_directories':self.checkpoints(path)}

    def test_resume_skips_past_stages_and_retains_five(self):
        path=OUTPUT/'resume'
        self.assertFalse(path.exists(),f"Refusing existing test output: {path}")
        first_recorder=SaveRecorder(stop_at=8)
        first=self.Trainer(model=self.Model(),args=self.args(path),train_dataset=self.Dataset(13),callbacks=[first_recorder])
        first.train()
        self.assertEqual(first_recorder.steps,[2,7])
        checkpoint=path/'checkpoint-7'
        before_mtime=(checkpoint/'trainer_state.json').stat().st_mtime_ns
        old_stage_mtime=(path/'checkpoint-2'/'trainer_state.json').stat().st_mtime_ns
        # Non-grid legacy checkpoints must lose rotation priority to planned
        # stages, even though their step numbers are newer than stage 2.
        (path/'checkpoint-3').mkdir();(path/'checkpoint-4').mkdir()
        second_recorder=SaveRecorder()
        resumed=self.Trainer(model=self.Model(),args=self.args(path),train_dataset=self.Dataset(13),callbacks=[second_recorder])
        resumed.train(resume_from_checkpoint=str(checkpoint))
        self.assertEqual(second_recorder.steps,[11,16,20])
        self.assertEqual(self.checkpoints(path),[2,7,11,16,20])
        self.assertEqual((checkpoint/'trainer_state.json').stat().st_mtime_ns,before_mtime)
        self.assertEqual((path/'checkpoint-2'/'trainer_state.json').stat().st_mtime_ns,old_stage_mtime)
        plan=json.loads((path/'checkpoint_schedule.json').read_text())
        self.assertEqual(plan['start_step'],7)
        self.assertEqual(plan['future_steps'],[11,16,20])
        self.assertEqual(plan['saved_this_run_steps'],[11,16,20])
        REPORT['integration']['resume']={'first_run_saved':first_recorder.steps,'resumed_from':7,
            'resumed_saved':second_recorder.steps,'final_checkpoints':self.checkpoints(path),
            'old_checkpoint_untouched':True,'non_grid_legacy_checkpoints_pruned':[3,4],
            'final_global_step':resumed.state.global_step}

    def test_disabled_preserves_periodic_saves(self):
        path=OUTPUT/'disabled'
        self.assertFalse(path.exists(),f"Refusing existing test output: {path}")
        args=self.args(path,max_steps=6,save_steps=2,save_total_limit=3,
                       save_first_fraction=None,save_num_checkpoints=None)
        recorder=SaveRecorder()
        trainer=self.Trainer(model=self.Model(),args=args,train_dataset=self.Dataset(13),callbacks=[recorder])
        trainer.train()
        self.assertEqual(recorder.steps,[2,4,6])
        self.assertEqual(self.checkpoints(path),[2,4,6])
        self.assertFalse((path/'checkpoint_schedule.json').exists())
        REPORT['integration']['disabled']={'save_steps':2,'saved_steps':recorder.steps,'checkpoint_directories':self.checkpoints(path)}


if __name__=='__main__':
    program=unittest.main(argv=[sys.argv[0]]+UNITTEST_ARGS,verbosity=2,exit=False)
    REPORT.update(tests_run=program.result.testsRun,failures=len(program.result.failures),
                  errors=len(program.result.errors),skipped=len(program.result.skipped),
                  success=program.result.wasSuccessful())
    (OUTPUT/'test_report.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print(json.dumps({'test_report':str(OUTPUT/'test_report.json'),'success':REPORT['success'],
                      'tests_run':REPORT['tests_run'],'integration':REPORT['integration']}),flush=True)
    raise SystemExit(0 if REPORT['success'] else 1)
