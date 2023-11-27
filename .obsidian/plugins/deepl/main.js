/*
THIS IS A GENERATED/BUNDLED FILE BY ESBUILD
if you want to view the source, please visit the github repository of this plugin
*/

var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/main.ts
var main_exports = {};
__export(main_exports, {
  default: () => DeepLPlugin
});
module.exports = __toCommonJS(main_exports);
var import_obsidian5 = require("obsidian");

// src/deepl/toLanguages.ts
var toLanguages = {
  BG: "Bulgarian",
  CS: "Czech",
  DA: "Danish",
  DE: "German",
  EL: "Greek",
  "EN-GB": "English (British)",
  "EN-US": "English (American)",
  ES: "Spanish",
  ET: "Estonian",
  FI: "Finnish",
  FR: "French",
  HU: "Hungarian",
  ID: "Indonesian",
  IT: "Italian",
  JA: "Japanese",
  LT: "Lithuanian",
  LV: "Latvian",
  NL: "Dutch",
  PL: "Polish",
  "PT-BR": "Portuguese (Brazilian)",
  "PT-PT": "Portuguese (European)",
  RO: "Romanian",
  RU: "Russian",
  SK: "Slovak",
  SL: "Slovenian",
  SV: "Swedish",
  TR: "Turkish",
  UK: "Ukrainian",
  ZH: "Chinese (simplified)"
};

// src/deepl/deeplException.ts
var DeepLException = class extends Error {
  constructor(code) {
    super();
    this.code = code;
  }
  static createFromStatusCode(statusCode, originalError) {
    const exception = new DeepLException(statusCode);
    exception.name = DeepLException.name;
    switch (statusCode) {
      case 403 /* FORBIDDEN */:
        exception.message = "Authentication failed. Please check your DeepL API key in the settings. Make sure you use the correct API free or pro.";
        break;
      case 413 /* PAYLOAD_TOO_LARGE */:
        exception.message = "Please try again with a shorter text";
        break;
      case 429 /* TOO_MANY_REQUEST */:
        exception.message = "You have done too many translations recently. Please try again later.";
        break;
      case 456 /* QUOTA_EXCEEDED */:
        exception.message = "The translation limit of your account has been reached. Consider upgrading your subscription.";
        break;
      default:
        exception.message = "An unknown error occured. See console for more details.";
        console.error(originalError, originalError.stack);
        break;
    }
    return exception;
  }
};

// src/deepl/deeplService.ts
var import_obsidian = require("obsidian");

// src/deepl/processor.ts
var Processor = class {
  constructor() {
    this.uniqueIDToLinkText = {};
  }
  preprocess(text) {
    let processedString = text;
    processedString = this.preprocessLinks(processedString);
    return processedString;
  }
  postprocess(text) {
    let processedString = text;
    processedString = this.postprocessLinks(processedString);
    return processedString;
  }
  preprocessLinks(text) {
    let processedText = text;
    this.uniqueIDToLinkText = {};
    const wikiLinkMatches = processedText.matchAll(/\[\[([^\]]+)\]\]/g);
    const markdownLinkMatches = processedText.matchAll(/(?:__|[*#])|\[.*?\]\((.*?)\)/g);
    const matches = [
      ...wikiLinkMatches || [],
      ...markdownLinkMatches || []
    ];
    for (const match of matches) {
      const uniqueString = this.generateUniqueString();
      this.uniqueIDToLinkText[uniqueString] = match[1];
      processedText = processedText.replace(match[1], uniqueString);
    }
    return processedText;
  }
  postprocessLinks(text) {
    let processedText = text;
    for (const [uniqueString, linkText] of Object.entries(this.uniqueIDToLinkText)) {
      processedText = processedText.replace(uniqueString, linkText);
    }
    return processedText;
  }
  generateUniqueString() {
    let uniqueString = "";
    const length = 16;
    const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    for (let i = 0; i < length; i++) {
      uniqueString += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    if (uniqueString in this.uniqueIDToLinkText) {
      return this.generateUniqueString();
    }
    return uniqueString;
  }
};

// src/deepl/deeplService.ts
var deeplFreeAPI = "https://api-free.deepl.com/v2";
var deeplProAPI = "https://api.deepl.com/v2";
var DeepLService = class {
  constructor(settings) {
    this.settings = settings;
    this.processor = new Processor();
  }
  async translate(text, toLanguage, fromLanguage) {
    var _a, _b;
    try {
      const preprocessedText = this.processor.preprocess(text);
      const useFromLanguage = fromLanguage != null && fromLanguage != "AUTO";
      const api = this.settings.useProAPI ? deeplProAPI : deeplFreeAPI;
      const response = await (0, import_obsidian.request)({
        url: `${api}/translate`,
        method: "POST",
        contentType: "application/x-www-form-urlencoded",
        body: new URLSearchParams({
          text: preprocessedText,
          target_lang: toLanguage,
          ...useFromLanguage && { source_lang: fromLanguage },
          formality: this.settings.formality
        }).toString(),
        headers: {
          Authorization: `DeepL-Auth-Key ${this.settings.apiKey}`
        },
        throw: true
      });
      const parsedResponse = JSON.parse(response);
      const translations = parsedResponse.translations;
      return translations.map((translation) => {
        return {
          text: this.processor.postprocess(translation.text),
          detected_source_language: translation.detected_source_language
        };
      });
    } catch (error) {
      let statusCode = 500;
      if (error instanceof Error) {
        statusCode = (_b = Number((_a = error.message.match("[0-9]{3}")) == null ? void 0 : _a[0])) != null ? _b : statusCode;
      }
      throw DeepLException.createFromStatusCode(statusCode, error);
    }
  }
};

// src/settings/pluginSettings.ts
var formalities = {
  default: "Default",
  prefer_more: "More formal",
  prefer_less: "Less formal"
};
var defaultSettings = {
  apiKey: "",
  fromLanguage: "AUTO",
  toLanguage: "DE",
  showStatusBar: true,
  useProAPI: false,
  formality: "default"
};

// src/settings/settingTab.ts
var import_obsidian2 = require("obsidian");

// src/deepl/fromLanguages.ts
var fromLanguages = {
  AUTO: "Detect language",
  BG: "Bulgarian",
  CS: "Czech",
  DA: "Danish",
  DE: "German",
  EL: "Greek",
  EN: "English",
  ES: "Spanish",
  ET: "Estonian",
  FI: "Finnish",
  FR: "French",
  HU: "Hungarian",
  ID: "Indonesian",
  IT: "Italian",
  JA: "Japanese",
  LT: "Lithuanian",
  LV: "Latvian",
  NL: "Dutch",
  PL: "Polish",
  PT: "Portuguese",
  RO: "Romanian",
  RU: "Russian",
  SK: "Slovak",
  SL: "Slovenian",
  SV: "Swedish",
  TR: "Turkish",
  UK: "Ukrainian",
  ZH: "Chinese"
};

// src/settings/settingTab.ts
var SettingTab = class extends import_obsidian2.PluginSettingTab {
  constructor(plugin) {
    super(plugin.app, plugin);
    this.plugin = plugin;
  }
  display() {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h3", {
      text: "Obsidian DeepL - Settings"
    });
    containerEl.createEl("h4", {
      text: "Commands"
    });
    new import_obsidian2.Setting(containerEl).setName("Translate selection").setDesc('Translates the selected text from the "From language" to the "To language".');
    new import_obsidian2.Setting(containerEl).setName("Translate selection: to language").setDesc("The target language can be selected by suggestion modal. The target language can be selected by suggestion modal. The selection will be replaced by the translation.");
    new import_obsidian2.Setting(containerEl).setName("Translate selection: From a language to another").setDesc("The source and target languages can be selected by suggestion modal. The target language can be selected by suggestion modal. The selection will be replaced by the translation.");
    new import_obsidian2.Setting(containerEl).setName("Translate selection: To language and append to selection").setDesc("The target language can be selected by suggestion modal. The translation will be appended to the selection.");
    containerEl.createEl("h4", {
      text: "Language settings"
    });
    new import_obsidian2.Setting(containerEl).setName("From language").setDesc("The language from which to translate.").addDropdown((dropdown) => dropdown.addOptions(fromLanguages).setValue(this.plugin.settings.fromLanguage).onChange(async (value) => {
      this.plugin.settings.fromLanguage = value;
      await this.plugin.saveSettings();
    }));
    new import_obsidian2.Setting(containerEl).setName("To language").setDesc("The language to translate into.").addDropdown((dropdown) => dropdown.addOptions(toLanguages).setValue(this.plugin.settings.toLanguage).onChange(async (value) => {
      this.plugin.settings.toLanguage = value;
      await this.plugin.saveSettings();
    }));
    new import_obsidian2.Setting(containerEl).setName("Formality").setDesc("Sets whether the translated text should lean towards formal or informal language.").addDropdown((dropdown) => dropdown.addOptions(formalities).setValue(this.plugin.settings.formality).onChange(async (value) => {
      this.plugin.settings.formality = value;
      await this.plugin.saveSettings();
    }));
    new import_obsidian2.Setting(containerEl).setName("Show setting in status bar").setDesc('Select the "To language" in the status bar.').addToggle((toggle) => toggle.setValue(this.plugin.settings.showStatusBar).onChange(async (value) => {
      this.plugin.settings.showStatusBar = value;
      await this.plugin.saveSettings();
    }));
    containerEl.createEl("h4", {
      text: "Authentication settings"
    });
    new import_obsidian2.Setting(containerEl).setName("DeepL API Key").setDesc("Get one for free at https://deepl.com/pro.").addText((text) => text.setPlaceholder("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:xx").setValue(this.plugin.settings.apiKey).onChange(async (value) => {
      this.plugin.settings.apiKey = value;
      await this.plugin.saveSettings();
    }));
    new import_obsidian2.Setting(containerEl).setName("Use DeepL Pro API").setDesc("Whether to use the DeepL Pro API or the DeepL Free API.").addToggle((toggle) => toggle.setValue(this.plugin.settings.useProAPI).onChange(async (value) => {
      this.plugin.settings.useProAPI = value;
      await this.plugin.saveSettings();
    }));
  }
};

// src/settings/statusbar.ts
var import_obsidian3 = require("obsidian");
function addStatusBar(plugin) {
  plugin.statusBar = plugin.addStatusBarItem();
  if (plugin.settings.showStatusBar === false) {
    plugin.statusBar.hide();
  }
  plugin.statusBar.addClass("statusbar-deepl");
  plugin.statusBar.addClass("mod-clickable");
  const toLanguage = toLanguages[plugin.settings.toLanguage];
  plugin.statusBar.createEl("span", { text: `\u{1F310} ${toLanguage}` });
  plugin.statusBar.onClickEvent((e) => handleStatusBarClick(e, plugin));
}
function handleStatusBarClick(mouseEvent, plugin) {
  const menu = new import_obsidian3.Menu();
  const filteredToLanguages = Object.entries(toLanguages).filter(([key]) => key !== plugin.settings.toLanguage);
  for (const [key, value] of filteredToLanguages) {
    menu.addItem(function(item) {
      return item.setTitle(value).onClick(async () => {
        plugin.settings.toLanguage = key;
        await plugin.saveSettings();
      });
    });
  }
  menu.showAtMouseEvent(mouseEvent);
}

// src/deepl/translateModal.ts
var import_obsidian4 = require("obsidian");
var TranslateModal = class extends import_obsidian4.SuggestModal {
  constructor(app2, placeholder, languages, callback) {
    super(app2);
    this.languages = languages;
    this.callback = callback;
    this.setPlaceholder(placeholder);
  }
  getSuggestions(query) {
    if (query) {
      return this.languages.filter((language) => language.code.toLowerCase().includes(query.toLowerCase()) || language.name.toLowerCase().includes(query.toLowerCase()));
    }
    return this.languages;
  }
  renderSuggestion(language, el) {
    el.createEl("div", { text: `${language.name} (${language.code})` });
  }
  onChooseSuggestion(language) {
    this.callback(language);
  }
  onClose() {
    let { contentEl } = this;
    contentEl.empty();
  }
};

// src/main.ts
var DeepLPlugin = class extends import_obsidian5.Plugin {
  async onload() {
    await this.loadSettings();
    this.deeplService = new DeepLService(this.settings);
    this.addSettingTab(new SettingTab(this));
    addStatusBar(this);
    this.addCommand({
      id: "deepl-translate-selection",
      name: "Translate selection",
      editorCallback: async (editor) => {
        if (editor.getSelection() === "") {
          return;
        }
        try {
          const translation = await this.deeplService.translate(editor.getSelection(), this.settings.toLanguage, this.settings.fromLanguage);
          editor.replaceSelection(translation[0].text);
        } catch (error) {
          if (error instanceof DeepLException) {
            new import_obsidian5.Notice(error.message);
          } else {
            console.error(error, error.stack);
            new import_obsidian5.Notice("An unknown error occured. See console for details.");
          }
        }
      }
    });
    this.addCommand({
      id: "deepl-translate-selection-append",
      name: "Translate selection: To language and append to selection",
      editorCallback: async (editor) => {
        if (editor.getSelection() === "") {
          return;
        }
        const selection = editor.getSelection();
        new TranslateModal(app, "To", Object.entries(toLanguages).map(([code, name]) => ({
          code,
          name
        })), async (language) => {
          try {
            const translation = await this.deeplService.translate(selection, language.code, this.settings.fromLanguage);
            editor.replaceSelection(`${selection} ${translation[0].text}`);
          } catch (error) {
            if (error instanceof DeepLException) {
              new import_obsidian5.Notice(error.message);
            } else {
              console.error(error, error.stack);
              new import_obsidian5.Notice("An unknown error occured. See console for details.");
            }
          }
        }).open();
      }
    });
    this.addCommand({
      id: "deepl-translate-selection-to",
      name: "Translate selection: to language",
      editorCallback: async (editor) => {
        if (editor.getSelection() === "") {
          return;
        }
        new TranslateModal(app, "To", Object.entries(toLanguages).map(([code, name]) => ({
          code,
          name
        })), async (language) => {
          try {
            const translation = await this.deeplService.translate(editor.getSelection(), language.code, this.settings.fromLanguage);
            editor.replaceSelection(translation[0].text);
          } catch (error) {
            if (error instanceof DeepLException) {
              new import_obsidian5.Notice(error.message);
            } else {
              console.error(error, error.stack);
              new import_obsidian5.Notice("An unknown error occured. See console for details.");
            }
          }
        }).open();
      }
    });
    this.addCommand({
      id: "deepl-translate-selection-from-to",
      name: "Translate selection: From a language to another",
      editorCallback: async (editor) => {
        if (editor.getSelection() === "") {
          return;
        }
        new TranslateModal(app, "From", Object.entries(fromLanguages).map(([code, name]) => ({
          code,
          name
        })), async (from) => {
          new TranslateModal(app, "To", Object.entries(toLanguages).map(([code, name]) => ({
            code,
            name
          })), async (to) => {
            try {
              const translation = await this.deeplService.translate(editor.getSelection(), to.code, from.code);
              editor.replaceSelection(translation[0].text);
            } catch (error) {
              if (error instanceof DeepLException) {
                new import_obsidian5.Notice(error.message);
              } else {
                console.error(error, error.stack);
                new import_obsidian5.Notice("An unknown error occured. See console for details.");
              }
            }
          }).open();
        }).open();
      }
    });
  }
  async loadSettings() {
    this.settings = Object.assign({}, defaultSettings, await this.loadData());
  }
  async saveSettings() {
    await this.saveData(this.settings);
    if (this.settings.showStatusBar === true) {
      this.statusBar.show();
      this.statusBar.setText(`\u{1F310} ${toLanguages[this.settings.toLanguage]}`);
    } else {
      this.statusBar.hide();
    }
  }
};
