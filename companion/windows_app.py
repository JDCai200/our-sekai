"""Standalone desktop UI. The existing OpenSekai player is never patched."""
import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]
if not getattr(sys, 'frozen', False):
    sys.path.insert(0, str(ROOT / 'backend'))
from hybrid_pjsk.genelive_settings import FIELDS, PRESETS, preset_settings, validate
from companion.song_package import load_json, write_json, publish, export_zip, extract_existing


def storage():
    return Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'OurSekaiCompanion'


def worker_command(request):
    worker = ROOT / 'AutoChart/OurSekai.Worker.exe'
    if worker.is_file():
        return [str(worker), str(request)]
    if not getattr(sys, 'frozen', False):
        config = ROOT / 'backend/runtime.local.json'
        if config.is_file():
            return [load_json(config)['python'], '-s', '-X', 'utf8', str(ROOT / 'backend/worker.py'), str(request)]
    raise FileNotFoundError('离线生成组件缺失，请完整解压发布包，保留 AutoChart 目录。')


class App:
    def __init__(self, window):
        self.window = window
        self.home = storage()
        self.home.mkdir(parents=True, exist_ok=True)
        self.config_path = self.home / 'settings.json'
        self.config = load_json(self.config_path) if self.config_path.is_file() else {}
        self.output_directory = Path(self.config.get('output_directory',str(self.home/'Songs')))
        self.messages = queue.Queue()
        self.process = None
        self.job = None
        self.selected_manifest = None
        self.last_project = None
        self.busy = False
        window.title('Our Sekai 0.2 · OpenSekai 自动谱面工具')
        window.geometry('800x650')
        window.minsize(740, 570)
        window.option_add('*Font', ('Microsoft YaHei UI', 10))
        self.audio = tk.StringVar()
        self.title = tk.StringVar()
        self.difficulty = tk.StringVar(value=self.config.get('difficulty', 'EXPERT'))
        if self.difficulty.get() not in PRESETS:
            self.difficulty.set('EXPERT')
        self.status = tk.StringVar(value='选择难度和歌曲，生成后导入已安装的 OpenSekai 社区版。')
        self.percent = tk.DoubleVar(value=0)
        self.advanced_visible = False
        self.advanced = {}
        outer = ttk.Frame(window, padding=20)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer,text='Our Sekai',font=('Microsoft YaHei UI',22,'bold')).pack(anchor='w')
        ttk.Label(outer,text='本地自动写谱  ·  通过歌曲包连接 OpenSekai 社区版').pack(anchor='w',pady=(0,16))
        song = ttk.Frame(outer)
        song.pack(fill='x')
        ttk.Entry(song,textvariable=self.audio,state='readonly').pack(side='left',fill='x',expand=True)
        self.pick = ttk.Button(song,text='导入歌曲',command=self.select_song)
        self.pick.pack(side='left',padx=6)
        self.pick_package = ttk.Button(song,text='已有歌曲包',command=self.select_package)
        self.pick_package.pack(side='left')
        row = ttk.Frame(outer)
        row.pack(fill='x',pady=12)
        ttk.Label(row,text='歌曲名称').pack(side='left')
        ttk.Entry(row,textvariable=self.title,width=35).pack(side='left',padx=8,fill='x',expand=True)
        ttk.Label(row,text='难度').pack(side='left')
        self.difficulty_box = ttk.Combobox(row,textvariable=self.difficulty,values=list(PRESETS),state='readonly',width=10)
        self.difficulty_box.pack(side='left',padx=8)
        self.difficulty_box.bind('<<ComboboxSelected>>',lambda event:self.reset_preset())
        ttk.Label(outer,text='新歌曲默认添加 9 秒静音；已有歌曲包沿用原前置，不重复添加。').pack(anchor='w')
        self.toggle = ttk.Button(outer,text='展开高级设置',command=self.toggle_advanced)
        self.toggle.pack(anchor='w',pady=10)
        self.advanced_frame = ttk.Frame(outer)
        canvas = tk.Canvas(self.advanced_frame,height=225,highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.advanced_frame,orient='vertical',command=canvas.yview)
        fields = ttk.Frame(canvas)
        canvas.create_window((0,0),window=fields,anchor='nw')
        fields.bind('<Configure>',lambda event:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left',fill='both',expand=True)
        scrollbar.pack(side='right',fill='y')
        settings = preset_settings(self.difficulty.get())
        saved = self.config.get('parameters', {})
        try:
            settings = validate({**settings, **saved})
        except (ValueError,TypeError):
            pass
        extra = [('filler','前置静音（秒）',9),('crop_start','原曲裁剪起点（秒）',0),
                 ('crop_duration','裁剪时长（秒，0 为完整）',0),('preview_start','原曲试听起点（秒）',0)]
        for index, (key,label,hint) in enumerate(FIELDS):
            self.add_field(fields,index,key,label,settings[key],hint)
        for index, (key,label,value) in enumerate(extra,len(FIELDS)):
            self.add_field(fields,index,key,label,self.config.get(key,value),'按原曲时间填写，程序自动换算前置')
        self.advanced['phase'][2].configure(text='留空自动；填写原曲节拍起点，程序自动加前置')
        buttons = ttk.Frame(outer)
        buttons.pack(fill='x',pady=10)
        self.generate_button = ttk.Button(buttons,text='生成可游玩歌曲包',command=self.generate)
        self.generate_button.pack(side='left')
        self.cancel_button = ttk.Button(buttons,text='取消生成',command=self.cancel,state='disabled')
        self.cancel_button.pack(side='left',padx=8)
        self.export_button = ttk.Button(buttons,text='导出歌曲包',command=self.export,state='disabled')
        self.export_button.pack(side='left')
        self.install_button = ttk.Button(buttons,text='加入 OpenSekai 歌曲库',command=self.install,state='disabled')
        self.install_button.pack(side='left',padx=8)
        ttk.Progressbar(outer,variable=self.percent,maximum=100).pack(fill='x',pady=8)
        ttk.Label(outer,textvariable=self.status,wraplength=750).pack(anchor='w',fill='x')
        bottom = ttk.Frame(outer)
        bottom.pack(fill='x',pady=12)
        self.output_picker = ttk.Button(bottom,text='设置输出位置',command=self.select_output)
        self.output_picker.pack(side='left')
        ttk.Button(bottom,text='打开输出目录',command=self.open_output).pack(side='left',padx=8)
        ttk.Button(bottom,text='打开 OpenSekai',command=self.launch_player).pack(side='left',padx=8)
        ttk.Label(outer,text='工具内不提供谱面编辑。游玩、成绩、视频和游戏设置由已安装的 OpenSekai 提供。',wraplength=750).pack(anchor='w')
        self.output_label = ttk.Label(outer,text='输出位置：'+str(self.output_directory),wraplength=750)
        self.output_label.pack(anchor='w')
        window.protocol('WM_DELETE_WINDOW',self.close)
        window.after(200,self.poll)

    def select_output(self):
        if self.busy:return
        folder=filedialog.askdirectory(title='选择歌曲和 ZIP 的输出位置',initialdir=str(self.output_directory if self.output_directory.exists() else self.home))
        if folder:
            self.output_directory=Path(folder)
            self.config['output_directory']=str(self.output_directory)
            write_json(self.config_path,self.config)
            self.output_label.configure(text='输出位置：'+str(self.output_directory))

    def open_output(self):
        try:
            self.output_directory.mkdir(parents=True,exist_ok=True)
            os.startfile(str(self.output_directory))
        except OSError as error:messagebox.showerror('无法打开输出目录',str(error))

    def add_field(self,parent,index,key,label,value,hint):
        variable = tk.StringVar(value='' if value is None else str(value))
        ttk.Label(parent,text=label,width=26).grid(row=index,column=0,sticky='w',pady=3)
        entry = ttk.Entry(parent,textvariable=variable,width=12)
        entry.grid(row=index,column=1,sticky='w',padx=8)
        description = ttk.Label(parent,text=hint,wraplength=360)
        description.grid(row=index,column=2,sticky='w')
        self.advanced[key] = variable,entry,description

    def reset_preset(self):
        current = {key:self.advanced[key][0].get() for key in ('bpm','phase')}
        for key,value in preset_settings(self.difficulty.get(),current).items():
            self.advanced[key][0].set('' if value is None else str(value))

    def toggle_advanced(self):
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.pack(fill='x',after=self.toggle)
        else:
            self.advanced_frame.pack_forget()
        self.toggle.configure(text='收起高级设置' if self.advanced_visible else '展开高级设置')

    def set_existing(self,manifest=None):
        self.selected_manifest = manifest
        for key in ('filler','crop_start','crop_duration','preview_start'):
            self.advanced[key][1].configure(state='disabled' if manifest else 'normal')

    def select_song(self):
        path = filedialog.askopenfilename(filetypes=[('歌曲','*.mp3 *.wav *.ogg *.flac *.m4a'),('所有文件','*.*')])
        if path:
            self.set_existing()
            self.audio.set(path)
            self.title.set(Path(path).stem)

    def select_package(self):
        path = filedialog.askopenfilename(filetypes=[('OpenSekai 歌曲包','*.zip')])
        if not path:
            return
        try:
            manifest,audio = extract_existing(path,self.home/'Imported'/uuid.uuid4().hex)
            self.set_existing(manifest)
            self.audio.set(str(audio))
            self.title.set(load_json(manifest)['title'])
            self.status.set('沿用已有包的音频、前置、试听、封面和视频；将另存生成结果。')
        except Exception as error:
            messagebox.showerror('歌曲包导入失败',str(error))

    def generate(self):
        if self.busy:
            return
        try:
            audio = Path(self.audio.get())
            if not audio.is_file():
                raise ValueError('请先导入歌曲')
            values = validate({key:self.advanced[key][0].get() for key,_,_ in FIELDS})
            request = dict(schema=1,audio=str(audio.resolve()),difficulty=self.difficulty.get(),
                           title=self.title.get().strip() or audio.stem,parameters=values)
            from math import isfinite
            for key in ('filler','crop_start','crop_duration','preview_start'):
                value = float(self.advanced[key][0].get())
                if not isfinite(value) or value < 0 or (key=='filler' and value>120):
                    raise ValueError('无效的前置或裁剪参数')
                request[key] = value
            if self.selected_manifest:
                request['existing_manifest'] = str(self.selected_manifest)
            self.job = self.home/'Jobs'/uuid.uuid4().hex
            self.job.mkdir(parents=True)
            write_json(self.job/'request.json',request)
            command = worker_command(self.job/'request.json')
            self.config.update(difficulty=self.difficulty.get(),parameters=values,
                               **{key:request[key] for key in ('filler','crop_start','crop_duration','preview_start')})
            write_json(self.config_path,self.config)
            self.busy = True
            self.percent.set(0)
            self.status.set('正在启动本地生成组件…')
            for button in (self.generate_button,self.pick,self.pick_package,self.export_button,self.install_button):
                button.configure(state='disabled')
            self.difficulty_box.configure(state='disabled')
            self.cancel_button.configure(state='normal')
            threading.Thread(target=self.run_job,args=(command,request),daemon=True).start()
        except Exception as error:
            messagebox.showerror('无法生成',str(error))

    def run_job(self,command,request):
        try:
            with (self.job/'worker-console.log').open('w',encoding='utf-8') as log:
                self.process = subprocess.Popen(command,cwd=self.job,stdout=log,stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                code = self.process.wait()
            if (self.job/'cancel').exists():
                self.messages.put(('cancelled',None)); return
            response = load_json(self.job/'response.json')
            if code or not response.get('success'):
                raise RuntimeError(response.get('error','生成组件异常结束'))
            project = publish(self.job/'project',self.output_directory,request.get('existing_manifest'))
            output = project.with_suffix('.zip')
            export_zip(project,output)
            self.messages.put(('done',(project,output)))
        except Exception as error:
            self.messages.put(('error',str(error)))

    def poll(self):
        if self.busy and self.job and (self.job/'status.json').exists():
            try:
                state = load_json(self.job/'status.json')
                self.status.set(state['message'])
                self.percent.set(max(self.percent.get(),state['percent']))
            except (OSError,ValueError):
                pass
        try:
            kind,value = self.messages.get_nowait()
            self.busy = False
            self.process = None
            for button in (self.generate_button,self.pick,self.pick_package):
                button.configure(state='normal')
            self.difficulty_box.configure(state='readonly')
            self.cancel_button.configure(state='disabled')
            if kind == 'done':
                self.last_project,output = value
                self.export_button.configure(state='normal')
                self.install_button.configure(state='normal')
                self.percent.set(100)
                self.status.set('已生成：'+str(output)+'。在 OpenSekai 歌曲管理中导入 ZIP 即可游玩。')
            elif kind == 'cancelled':
                self.status.set('已取消生成。现有歌曲库保持原样。')
            else:
                self.status.set('生成失败：'+value+'；日志：'+str(self.job))
        except queue.Empty:
            pass
        self.window.after(200,self.poll)

    def cancel(self):
        if self.busy and self.job:
            (self.job/'cancel').touch()
            self.status.set('正在取消…')

    def export(self):
        if not self.last_project:
            return
        path = filedialog.asksaveasfilename(defaultextension='.zip',initialfile=self.last_project.name+'.zip',filetypes=[('歌曲包','*.zip')])
        if path:
            export_zip(self.last_project,path)
            self.status.set('歌曲包已导出：'+path)

    def install(self):
        default = Path.home()/'AppData/LocalLow/JsoftStudio/Ojsk Community/CustomMusicScores'
        folder = filedialog.askdirectory(title='选择 OpenSekai 的 CustomMusicScores 歌曲库目录',initialdir=str(default if default.exists() else Path.home()))
        if not folder:
            return
        root = Path(folder)
        if root.name != 'CustomMusicScores':
            messagebox.showerror('歌曲库目录不正确','请选择 OpenSekai 的 CustomMusicScores 目录。'); return
        destination = root/self.last_project.name
        if destination.exists():
            messagebox.showinfo('歌曲已存在','该歌曲已加入，未覆盖。'); return
        temporary = root/('.import-'+uuid.uuid4().hex)
        try:
            shutil.copytree(self.last_project,temporary)
            os.replace(temporary,destination)
            self.status.set('已加入歌曲库。请在 OpenSekai 中刷新歌曲列表。')
        except Exception as error:
            messagebox.showerror('加入失败',str(error))

    def launch_player(self):
        path = self.config.get('player_exe')
        if not path or not Path(path).is_file():
            path = filedialog.askopenfilename(title='选择已安装的 OpenSekai 游戏 EXE',filetypes=[('Windows 程序','*.exe')])
        if path:
            self.config['player_exe'] = path
            write_json(self.config_path,self.config)
            subprocess.Popen([path],cwd=Path(path).parent)

    def close(self):
        self.cancel()
        self.window.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test',type=Path)
    parser.add_argument('--generate',type=Path,help='Integration-check request file; normal users use the window')
    parser.add_argument('--output-directory',type=Path)
    parser.add_argument('--songs-directory',type=Path,help='Publish generated assets to a user-selected folder')
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    if args.self_test:
        from companion.song_package import score_from_sus
        fixture = '#00002: 4\n#BPM01: 120\n#00008: 01\n#00012: 13\n'
        assert len(score_from_sus(fixture)['NoteList']) == 1
        assert validate(preset_settings('MASTER'))['snap_division'] == 32
        tk.Tcl().eval('info patchlevel')
        write_json(args.self_test,dict(desktop_ui_modules_passed=True,tk_runtime_available=True,
            worker_found=(ROOT/'AutoChart/OurSekai.Worker.exe').is_file()))
        return
    if args.generate:
        import traceback
        home=args.output_directory or storage()
        job=home/'Jobs'/uuid.uuid4().hex; job.mkdir(parents=True)
        request=load_json(args.generate); write_json(job/'request.json',request)
        try:
            with (job/'worker-console.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(worker_command(job/'request.json'),cwd=job,stdout=log,stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            response=load_json(job/'response.json')
            if result.returncode or not response.get('success'):raise RuntimeError(response.get('error','Worker failed'))
            project=publish(job/'project',args.songs_directory or home/'Songs',request.get('existing_manifest'))
            output=project.with_suffix('.zip'); export_zip(project,output)
            report=dict(success=True,project=str(project),package=str(output),note_count=response['note_count'])
        except Exception as error:
            report=dict(success=False,error=str(error),traceback=traceback.format_exc())
        if args.report:write_json(args.report,report)
        raise SystemExit(0 if report['success'] else 1)
    window = tk.Tk()
    App(window)
    window.mainloop()


if __name__ == '__main__':
    main()
