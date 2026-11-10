"use client";

import { Eye, EyeOff, LoaderCircle, Save, Server, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { fetchExternalUpstream, updateExternalUpstream, type ExternalUpstreamSettings } from "@/lib/api";

export function ExternalUpstreamCard() {
  const [settings, setSettings] = useState<ExternalUpstreamSettings | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [clearKey, setClearKey] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void fetchExternalUpstream()
      .then((data) => setSettings(data.upstream))
      .catch((error) => toast.error(error instanceof Error ? error.message : "加载模型上游失败"))
      .finally(() => setLoading(false));
  }, []);

  async function save() {
    if (!settings) return;
    setSaving(true);
    try {
      const data = await updateExternalUpstream({
        enabled: settings.enabled,
        base_url: settings.base_url.trim(),
        api_key: apiKey.trim(),
        clear_api_key: clearKey,
      });
      setSettings(data.upstream);
      setApiKey("");
      setClearKey(false);
      setShowKey(false);
      toast.success("模型上游配置已保存");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "保存模型上游失败");
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card className="rounded-2xl border-white/80 bg-white/90 shadow-sm">
      <CardContent className="space-y-5 p-6">
        <div className="flex items-center gap-3">
          <Server className="size-5 text-stone-500" />
          <h2 className="text-base font-semibold text-stone-900">模型上游</h2>
        </div>
        {loading || !settings ? (
          <div className="flex justify-center py-10"><LoaderCircle className="size-5 animate-spin text-stone-400" /></div>
        ) : (
          <>
            <label className="flex items-center gap-3 text-sm text-stone-700">
              <Checkbox
                checked={settings.enabled}
                onCheckedChange={(checked) => setSettings({ ...settings, enabled: Boolean(checked) })}
              />
              启用外部模型上游
            </label>
            <p className="text-sm leading-6 text-stone-500">
              本地账号池（包括导入的 Sub2API 账号）会继续提供额度。模型目录不额外显示 upstream/ 前缀选项；图片生成优先走外部上游，失败时再回退到本地支持的图片模型。
            </p>
            <div className="space-y-2">
              <label htmlFor="external-upstream-url" className="text-sm font-medium text-stone-700">服务地址</label>
              <Input
                id="external-upstream-url"
                value={settings.base_url}
                onChange={(event) => setSettings({ ...settings, base_url: event.target.value })}
                placeholder="https://daixu.qzz.io/"
                className="h-10 rounded-xl border-stone-200 bg-white"
              />
              <p className="text-sm leading-6 text-stone-500">
                可填写服务根地址或带 /v1 的地址，系统会自动处理接口路径。
              </p>
            </div>
            <div className="space-y-2">
              <label htmlFor="external-upstream-key" className="text-sm font-medium text-stone-700">
                上游 API Key {settings.has_api_key && !clearKey ? "(已保存)" : ""}
              </label>
              <div className="flex gap-2">
                <Input
                  id="external-upstream-key"
                  type={showKey ? "text" : "password"}
                  value={apiKey}
                  onChange={(event) => { setApiKey(event.target.value); setClearKey(false); }}
                  placeholder={settings.has_api_key && !clearKey ? "留空保持现有 Key" : "输入上游 API Key"}
                  autoComplete="off"
                  className="h-10 rounded-xl border-stone-200 bg-white"
                />
                <Button type="button" variant="outline" size="icon" aria-label={showKey ? "隐藏 Key" : "显示 Key"} title={showKey ? "隐藏 Key" : "显示 Key"} onClick={() => setShowKey(!showKey)}>
                  {showKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                </Button>
                {settings.has_api_key && (
                  <Button type="button" variant="outline" size="icon" aria-label="清除已保存的 Key" title="清除已保存的 Key" onClick={() => { setClearKey(true); setApiKey(""); setSettings({ ...settings, enabled: false }); }}>
                    <Trash2 className="size-4" />
                  </Button>
                )}
              </div>
            </div>
            <div className="flex justify-end">
              <Button className="h-10 rounded-xl bg-stone-950 px-5 text-white hover:bg-stone-800" onClick={() => void save()} disabled={saving}>
                {saving ? <LoaderCircle className="size-4 animate-spin" /> : <Save className="size-4" />}
                保存
              </Button>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
