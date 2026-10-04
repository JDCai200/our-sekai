package org.oursekai.companion;

import android.app.*;
import android.content.*;
import android.content.pm.ResolveInfo;
import android.database.Cursor;
import android.net.Uri;
import android.os.*;
import android.provider.OpenableColumns;
import android.provider.DocumentsContract;
import android.view.View;
import android.widget.*;
import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Native companion UI. Player package is exported through Android SAF. */
public final class MainActivity extends Activity {
    private static final int AUDIO=10, PACKAGE=11, EXPORT=12, OUTPUT_DIRECTORY=13;
    private String audio,existing;
    private EditText title;
    private Spinner difficulty;
    private LinearLayout advanced;
    private final Map<String,EditText> fields = new LinkedHashMap<>();
    private TextView status,selected;
    private ProgressBar progress;
    private Button chooseAudio,choosePackage,generate,export,cancel,chooseOutput;
    private TextView outputLocation;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private boolean destroyed;
    private volatile boolean importing;
    private String displayedResultJob;
    private boolean difficultyInitialized;

    private Button button(LinearLayout parent,String text,View.OnClickListener callback) {
        Button button = new Button(this); button.setText(text); button.setOnClickListener(callback); parent.addView(button); return button;
    }
    private EditText field(LinearLayout parent,String label,String key,String value) {
        TextView text=new TextView(this); text.setText(label); parent.addView(text);
        EditText input=new EditText(this); input.setSingleLine(); input.setText(value); parent.addView(input); fields.put(key,input); return input;
    }
    private void message(String text) { status.setText(text); }

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        ScrollView scroll = new ScrollView(this);
        LinearLayout outer = new LinearLayout(this); outer.setOrientation(LinearLayout.VERTICAL); outer.setPadding(28,24,28,24);
        scroll.addView(outer); setContentView(scroll);
        TextView heading=new TextView(this); heading.setText("Our Sekai"); heading.setTextSize(28); outer.addView(heading);
        TextView description=new TextView(this); description.setText("OpenSekai 社区版的离线自动谱面工具\n手机本地生成，无需电脑或服务器"); outer.addView(description);
        difficulty=new Spinner(this); difficulty.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"EASY","NORMAL","HARD","EXPERT","MASTER"}));
        outer.addView(difficulty); difficulty.setSelection(3);
        chooseAudio=button(outer,"导入歌曲",view -> pick(AUDIO,"audio/*"));
        choosePackage=button(outer,"导入已有歌曲包并重新生成",view -> pick(PACKAGE,"application/zip"));
        selected=new TextView(this); selected.setText("尚未选择歌曲"); outer.addView(selected);
        TextView nameLabel=new TextView(this); nameLabel.setText("歌曲名称"); outer.addView(nameLabel);
        title=new EditText(this); title.setSingleLine(); outer.addView(title);
        TextView padding=new TextView(this); padding.setText("新歌曲默认实际添加 9 秒静音；已有歌曲包不会重复添加。"); outer.addView(padding);
        advanced=new LinearLayout(this); advanced.setOrientation(LinearLayout.VERTICAL); advanced.setVisibility(View.GONE);
        button(outer,"高级设置",view -> advanced.setVisibility(advanced.getVisibility()==View.GONE?View.VISIBLE:View.GONE)); outer.addView(advanced);
        field(advanced,"模型难度条件（10 / 20 / 30 / 40 / 50）","condition","40");
        field(advanced,"采音阈值","threshold","0.4"); field(advanced,"最小间距（帧）","min_distance","4");
        field(advanced,"时间偏移（毫秒）","shift_ms","0"); field(advanced,"节奏吸附上限（毫秒）","snap_ms","0");
        field(advanced,"全点网格（0 / 8 / 16 / 32）","snap_division","16"); field(advanced,"BPM（0 自动）","bpm","0");
        field(advanced,"原曲节拍起点（秒，留空自动）","phase",""); field(advanced,"提取方式（upstream / peaks）","method","upstream");
        field(advanced,"前置静音（秒）","filler","9"); field(advanced,"原曲裁剪起点（秒）","crop_start","0");
        field(advanced,"裁剪时长（秒，0 完整）","crop_duration","0"); field(advanced,"原曲试听起点（秒）","preview_start","0");
        difficulty.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            public void onNothingSelected(android.widget.AdapterView<?> parent) {}
            public void onItemSelected(android.widget.AdapterView<?> parent,View view,int position,long id) {
                if (!difficultyInitialized) { difficultyInitialized=true; return; }
                fields.get("condition").setText(Integer.toString((position+1)*10)); fields.get("threshold").setText(position<2?"0.25":"0.4");
                fields.get("min_distance").setText(position==0?"5":"4"); fields.get("snap_division").setText(position==4?"32":"16");
            }
        });
        generate=button(outer,"生成可游玩歌曲包",view -> generate());
        chooseOutput=button(outer,"设置输出文件夹（生成后自动保存 ZIP）",view -> {
            Intent intent=new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION|Intent.FLAG_GRANT_WRITE_URI_PERMISSION|Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
            startActivityForResult(intent,OUTPUT_DIRECTORY);
        });
        outputLocation=new TextView(this); outer.addView(outputLocation); showOutputLocation();
        cancel=button(outer,"取消生成",view -> startService(new Intent(this,GenerationService.class).setAction("cancel")));
        export=button(outer,"保存歌曲包，然后在 OpenSekai 中导入",view -> export());
        button(outer,"打开已安装的 OpenSekai",view -> launchPlayer());
        progress=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal); progress.setMax(100); outer.addView(progress);
        status=new TextView(this); outer.addView(status); message("选择难度和歌曲开始生成。");
        TextView footer=new TextView(this); footer.setText("工具不提供谱面编辑。游玩、视频、成绩和游戏设置由 OpenSekai 提供。\n建议使用 Android 7+ 的 ARM64 手机；当前 native 数值库针对 4 KB 页面设备。"); outer.addView(footer);
        restore(); handler.post(this::poll);
        if (Build.VERSION.SDK_INT>=33) requestPermissions(new String[]{"android.permission.POST_NOTIFICATIONS"},20);
    }

    private void pick(int code,String mime) {
        Intent intent=new Intent(Intent.ACTION_OPEN_DOCUMENT).setType(mime).addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(intent,code);
    }
    private String filename(Uri uri) {
        try (Cursor cursor=getContentResolver().query(uri,new String[]{OpenableColumns.DISPLAY_NAME},null,null,null)) {
            if (cursor!=null && cursor.moveToFirst()) return cursor.getString(0);
        }
        return "song";
    }
    private void copy(Uri uri,File target) throws IOException {
        target.getParentFile().mkdirs();
        try (InputStream input=getContentResolver().openInputStream(uri); OutputStream output=new FileOutputStream(target)) {
            if (input==null) throw new IOException("无法读取所选文件");
            byte[] buffer=new byte[1024*1024]; int size;
            while ((size=input.read(buffer))!=-1) output.write(buffer,0,size);
        }
    }
    @Override protected void onActivityResult(int requestCode,int resultCode,Intent data) {
        super.onActivityResult(requestCode,resultCode,data);
        if (resultCode!=RESULT_OK || data==null || data.getData()==null) return;
        Uri uri=data.getData();
        if (requestCode==OUTPUT_DIRECTORY) {
            try {
                int flags=data.getFlags() & (Intent.FLAG_GRANT_READ_URI_PERMISSION|Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
                if ((flags & Intent.FLAG_GRANT_WRITE_URI_PERMISSION)==0) throw new IOException("所选位置没有写入权限");
                getContentResolver().takePersistableUriPermission(uri,flags);
                getSharedPreferences("state",MODE_PRIVATE).edit().putString("output_tree",uri.toString()).apply();
                showOutputLocation(); message("输出文件夹已保存。后续生成成功后自动保存 ZIP；原有文件不会覆盖。");
            } catch (Exception error) { message("设置输出位置失败："+error.getMessage()); }
            return;
        }
        if (requestCode==EXPORT) {
            String path=getSharedPreferences("state",MODE_PRIVATE).getString("zip",null);
            new Thread(() -> {
                try (InputStream input=new FileInputStream(path); OutputStream output=getContentResolver().openOutputStream(uri,"wt")) {
                    if (output==null) throw new IOException("无法保存歌曲包");
                    byte[] buffer=new byte[1024*1024]; int size;
                    while ((size=input.read(buffer))!=-1) output.write(buffer,0,size);
                    runOnUiThread(() -> message("歌曲包已保存。在 OpenSekai 的歌曲管理中导入该 ZIP，即可进入 Live。"));
                } catch (Exception error) { runOnUiThread(() -> message("保存失败："+error.getMessage())); }
            }).start(); return;
        }
        String name=filename(uri); String suffix=name.contains(".")?name.substring(name.lastIndexOf('.')).replaceAll("[^a-zA-Z0-9.]",""):".audio";
        File target=new File(getFilesDir(),"Imported/"+UUID.randomUUID()+suffix);
        importing=true;
        message("正在读取文件…");
        new Thread(() -> {
            try {
                copy(uri,target);
                if (requestCode==PACKAGE) {
                    String result=GenerationService.python(this).getModule("portable.android_bridge").callAttr("inspect_package",target.getAbsolutePath(),
                        new File(getFilesDir(),"Imported/"+UUID.randomUUID()).getAbsolutePath()).toString();
                    JSONObject packageInfo=new JSONObject(result);
                    audio=packageInfo.getString("audio"); existing=packageInfo.getString("manifest");
                    runOnUiThread(() -> { title.setText(packageInfo.optString("title")); selected.setText(name); setExisting(); message("已导入歌曲包，将沿用原前置并另存新谱面。"); save(); });
                } else {
                    audio=target.getAbsolutePath(); existing=null;
                    runOnUiThread(() -> { title.setText(name.contains(".")?name.substring(0,name.lastIndexOf('.')):name); selected.setText(name); setExisting(); message("歌曲已选择。"); save(); });
                }
            } catch (Exception error) { runOnUiThread(() -> message("导入失败："+error.getMessage())); }
            finally { importing=false; }
        }).start();
    }

    private void setExisting() {
        for (String key:new String[]{"filler","crop_start","crop_duration","preview_start"}) fields.get(key).setEnabled(existing==null);
    }
    private JSONObject request() throws JSONException {
        JSONObject request=new JSONObject().put("schema",1).put("audio",audio).put("title",title.getText().toString()).put("difficulty",difficulty.getSelectedItem().toString());
        JSONObject parameters=new JSONObject();
        for (Map.Entry<String,EditText> row:fields.entrySet()) {
            String value=row.getValue().getText().toString().trim(); String key=row.getKey();
            if (key.equals("filler")||key.equals("crop_start")||key.equals("crop_duration")||key.equals("preview_start")) request.put(key,value);
            else parameters.put(key,key.equals("phase")&&value.isEmpty()?JSONObject.NULL:value);
        }
        request.put("parameters",parameters);
        if (existing!=null) request.put("existing_manifest",existing);
        return request;
    }
    private void generate() {
        if (audio==null || !new File(audio).isFile()) { message("请先导入歌曲。"); return; }
        try {
            JSONObject request=request(); save();
            Intent intent=new Intent(this,GenerationService.class).setAction("generate").putExtra("request",request.toString());
            if (Build.VERSION.SDK_INT>=26) startForegroundService(intent); else startService(intent);
            message("正在启动本地生成任务…");
        } catch (Exception error) { message("无法生成："+error.getMessage()); }
    }
    private void export() {
        String path=getSharedPreferences("state",MODE_PRIVATE).getString("zip",null);
        if (path==null || !new File(path).isFile()) { message("请先生成歌曲包。"); return; }
        String tree=getSharedPreferences("state",MODE_PRIVATE).getString("output_tree",null);
        if (tree!=null) { saveToDirectory(path,tree); return; }
        Intent intent=new Intent(Intent.ACTION_CREATE_DOCUMENT).setType("application/zip").addCategory(Intent.CATEGORY_OPENABLE)
            .putExtra(Intent.EXTRA_TITLE,new File(path).getName()); startActivityForResult(intent,EXPORT);
    }
    private void showOutputLocation() {
        String tree=getSharedPreferences("state",MODE_PRIVATE).getString("output_tree",null);
        outputLocation.setText(tree==null?"输出位置：每次保存时选择":"输出文件夹："+Uri.decode(tree.substring(tree.lastIndexOf('/')+1)));
    }
    private void saveToDirectory(String path,String treeText) {
        message("正在保存到设置的输出文件夹…");
        new Thread(() -> {
            Uri document=null;
            try {
                Uri tree=Uri.parse(treeText);
                Uri parent=DocumentsContract.buildDocumentUriUsingTree(tree,DocumentsContract.getTreeDocumentId(tree));
                document=DocumentsContract.createDocument(getContentResolver(),parent,"application/zip",new File(path).getName());
                if (document==null) throw new IOException("无法在所选文件夹创建 ZIP");
                try (InputStream input=new FileInputStream(path); OutputStream output=getContentResolver().openOutputStream(document,"w")) {
                    if (output==null) throw new IOException("输出位置无法写入");
                    byte[] buffer=new byte[1024*1024]; int size;
                    while ((size=input.read(buffer))!=-1) output.write(buffer,0,size);
                }
                runOnUiThread(() -> message("歌曲包已保存到设置的文件夹，请在 OpenSekai 中导入 ZIP。"));
            } catch (Exception error) {
                if (document!=null) try { DocumentsContract.deleteDocument(getContentResolver(),document); } catch (Exception ignored) {}
                runOnUiThread(() -> message("保存失败："+error.getMessage()+"；可重新设置输出文件夹后点击保存，已生成的歌曲包仍保留。"));
            }
        },"OurSekai-Export").start();
    }
    private void save() {
        try { getSharedPreferences("state",MODE_PRIVATE).edit().putString("settings",request().toString()).putString("title",title.getText().toString()).apply(); }
        catch (Exception ignored) {}
    }
    private void restore() {
        try {
            JSONObject saved=new JSONObject(getSharedPreferences("state",MODE_PRIVATE).getString("settings","{}"));
            audio=saved.optString("audio",null); existing=saved.optString("existing_manifest",null);
            title.setText(saved.optString("title","")); selected.setText(audio==null?"尚未选择歌曲":title.getText());
            String[] names={"EASY","NORMAL","HARD","EXPERT","MASTER"};
            int index=Arrays.asList(names).indexOf(saved.optString("difficulty","EXPERT")); difficulty.setSelection(index<0?3:index);
            JSONObject values=saved.optJSONObject("parameters");
            for (String key:fields.keySet()) {
                if (saved.has(key)) fields.get(key).setText(saved.getString(key));
                else if (values!=null && values.has(key)) fields.get(key).setText(values.isNull(key)?"":values.getString(key));
            }
            setExisting();
        } catch (Exception ignored) {}
    }
    private void poll() {
        if (destroyed) return;
        boolean running=GenerationService.running;
        generate.setEnabled(!running&&!importing); chooseAudio.setEnabled(!running&&!importing); choosePackage.setEnabled(!running&&!importing); difficulty.setEnabled(!running);
        cancel.setEnabled(running); export.setEnabled(!running && getSharedPreferences("state",MODE_PRIVATE).contains("zip"));
        chooseOutput.setEnabled(!running&&!importing);
        String job=getSharedPreferences("state",MODE_PRIVATE).getString("job",null);
        if (job!=null) {
            File file=new File(job,running?"status.json":"result.json");
            if (file.isFile() && (running || !job.equals(displayedResultJob))) try {
                ByteArrayOutputStream bytes=new ByteArrayOutputStream();
                try (InputStream input=new FileInputStream(file)) { byte[] buffer=new byte[4096]; int size; while ((size=input.read(buffer))!=-1) bytes.write(buffer,0,size); }
                String json=new String(bytes.toByteArray(),StandardCharsets.UTF_8);
                JSONObject state=new JSONObject(json); message(state.optString("message"));
                progress.setProgress(state.optInt("percent",state.optBoolean("success")?100:0));
                if (!running) {
                    displayedResultJob=job;
                    String tree=getSharedPreferences("state",MODE_PRIVATE).getString("output_tree",null);
                    String exported=getSharedPreferences("state",MODE_PRIVATE).getString("exported_job",null);
                    if (state.optBoolean("success") && tree!=null && !job.equals(exported)) {
                        getSharedPreferences("state",MODE_PRIVATE).edit().putString("exported_job",job).apply();
                        saveToDirectory(state.getString("zip"),tree);
                    }
                }
            } catch (Exception ignored) {}
        }
        handler.postDelayed(this::poll,500);
    }
    private void launchPlayer() {
        Intent query=new Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER);
        List<ResolveInfo> players=new ArrayList<>(); List<String> labels=new ArrayList<>();
        for (ResolveInfo info:getPackageManager().queryIntentActivities(query,0)) {
            String name=info.loadLabel(getPackageManager()).toString(); String lower=name.toLowerCase(Locale.ROOT);
            if (!info.activityInfo.packageName.equals(getPackageName()) && (lower.contains("opensekai")||lower.contains("ojsk"))) { players.add(info); labels.add(name); }
        }
        if (players.isEmpty()) { message("未找到 OpenSekai 社区版，请从手机桌面打开已安装的游戏。"); return; }
        new AlertDialog.Builder(this).setTitle("打开 OpenSekai").setItems(labels.toArray(new String[0]),(dialog,index) -> {
            ResolveInfo info=players.get(index); startActivity(new Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
                .setComponent(new ComponentName(info.activityInfo.packageName,info.activityInfo.name)));
        }).show();
    }
    @Override protected void onDestroy() { destroyed=true; handler.removeCallbacksAndMessages(null); super.onDestroy(); }
}
