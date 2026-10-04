"""Android service entry points. Files are private until a validated export."""
import json
from pathlib import Path
import traceback
from companion.song_package import load_json,write_json,extract_existing,publish,export_zip
from .pipeline import run,verify_models


def inspect_package(package,destination):
    manifest,audio=extract_existing(package,destination)
    return json.dumps(dict(manifest=str(manifest),audio=str(audio),title=load_json(manifest)['title']),ensure_ascii=False)


def generate(request_text,models,job,home):
    job=Path(job); home=Path(home); request=json.loads(request_text)
    def progress(message,percent):
        status=job/'status.json'; temporary=job/'status.tmp'
        write_json(temporary,dict(message=message,percent=percent)); temporary.replace(status)
    try:
        stage=run(request,models,job,android=True,progress=progress)
        if (job/'cancel').exists():raise InterruptedError('已取消生成')
        project=publish(stage,home/'Songs',request.get('existing_manifest'))
        output=project.with_suffix('.zip'); export_zip(project,output)
        result=dict(success=True,zip=str(output),project=str(project),message='歌曲包已生成，请保存后在 OpenSekai 中导入。')
    except Exception as error:
        (job/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        result=dict(success=False,cancelled=(job/'cancel').exists(),message=str(error))
    write_json(job/'result.json',result)
    return json.dumps(result,ensure_ascii=False)
