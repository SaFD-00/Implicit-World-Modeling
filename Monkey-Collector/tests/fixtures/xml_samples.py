"""XML sample constants for testing."""

# Minimal XML — only invisible/zero-bounds nodes → empty parse result
MINIMAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="" class="android.widget.FrameLayout"
        content-desc="" checkable="false" checked="false" clickable="false"
        enabled="true" focusable="false" focused="false" scrollable="false"
        long-clickable="false" password="false" selected="false"
        bounds="[0,0][0,0]" package="com.test.app"
        visible-to-user="false" important="false" />
</hierarchy>
"""

# Simple XML — 6 visible nodes with various interactable types
SIMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="" class="android.widget.FrameLayout"
        content-desc="" checkable="false" checked="false" clickable="false"
        enabled="true" focusable="false" focused="false" scrollable="false"
        long-clickable="false" password="false" selected="false"
        bounds="[0,0][1080,1920]" package="com.test.app"
        visible-to-user="true" important="false">
    <node index="1" text="" resource-id="com.test.app:id/toolbar"
          class="android.view.ViewGroup"
          content-desc="" checkable="false" checked="false" clickable="false"
          enabled="true" focusable="false" focused="false" scrollable="false"
          long-clickable="false" password="false" selected="false"
          bounds="[0,0][1080,168]" package="com.test.app"
          visible-to-user="true" important="false">
      <node index="2" text="" resource-id="com.test.app:id/search_btn"
            class="android.widget.ImageButton"
            content-desc="Search" checkable="false" checked="false"
            clickable="true" enabled="true" focusable="true" focused="false"
            scrollable="false" long-clickable="false" password="false"
            selected="false" bounds="[900,24][1056,144]" package="com.test.app"
            visible-to-user="true" important="true" />
    </node>
    <node index="3" text="" resource-id="com.test.app:id/search_input"
          class="android.widget.EditText"
          content-desc="Search field" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false"
          selected="false" bounds="[24,200][1056,300]" package="com.test.app"
          visible-to-user="true" important="true" />
    <node index="4" text="" resource-id="com.test.app:id/list_container"
          class="android.widget.ScrollView"
          content-desc="" checkable="false" checked="false" clickable="false"
          enabled="true" focusable="false" focused="false" scrollable="true"
          long-clickable="false" password="false" selected="false"
          bounds="[0,300][1080,1800]" package="com.test.app"
          visible-to-user="true" important="false">
      <node index="5" text="Item title" resource-id="com.test.app:id/item_title"
            class="android.widget.TextView"
            content-desc="" checkable="false" checked="false" clickable="false"
            enabled="true" focusable="false" focused="false" scrollable="false"
            long-clickable="false" password="false" selected="false"
            bounds="[24,320][1056,420]" package="com.test.app"
            visible-to-user="true" important="false" />
    </node>
    <node index="6" text="" resource-id="com.test.app:id/fab"
          class="android.widget.Button"
          content-desc="Add new" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="true" password="false"
          selected="false" bounds="[900,1680][1056,1800]" package="com.test.app"
          visible-to-user="true" important="true" />
  </node>
</hierarchy>
"""

# Input-only XML — the sole actionable node is a text field (EditText → SET_TEXT).
# Exercises the fallback demotion's "not exclusion" branch: with no non-input
# element available, the input must still be selectable so the screen yields an
# action instead of an illegal root back-press.
INPUT_ONLY_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="" class="android.widget.FrameLayout"
        content-desc="" checkable="false" checked="false" clickable="false"
        enabled="true" focusable="false" focused="false" scrollable="false"
        long-clickable="false" password="false" selected="false"
        bounds="[0,0][1080,1920]" package="com.test.app"
        visible-to-user="true" important="false">
    <node index="1" text="Search music" resource-id="com.test.app:id/hint"
          class="android.widget.TextView"
          content-desc="" checkable="false" checked="false" clickable="false"
          enabled="true" focusable="false" focused="false" scrollable="false"
          long-clickable="false" password="false" selected="false"
          bounds="[24,80][1056,160]" package="com.test.app"
          visible-to-user="true" important="false" />
    <node index="2" text="" resource-id="com.test.app:id/search_input"
          class="android.widget.EditText"
          content-desc="Search field" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false"
          selected="false" bounds="[24,200][1056,300]" package="com.test.app"
          visible-to-user="true" important="true" />
  </node>
</hierarchy>
"""

# Complex XML — ~12 visible nodes with checkable, long-clickable, nested structures
COMPLEX_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="" class="android.widget.FrameLayout"
        content-desc="" checkable="false" checked="false" clickable="false"
        enabled="true" focusable="false" focused="false" scrollable="false"
        long-clickable="false" password="false" selected="false"
        bounds="[0,0][1080,1920]" package="com.test.app"
        visible-to-user="true" important="false">
    <node index="1" text="" resource-id="com.test.app:id/toolbar"
          class="android.view.ViewGroup"
          content-desc="" checkable="false" checked="false" clickable="false"
          enabled="true" focusable="false" focused="false" scrollable="false"
          long-clickable="false" password="false" selected="false"
          bounds="[0,0][1080,168]" package="com.test.app"
          visible-to-user="true" important="false">
      <node index="2" text="Settings" resource-id="com.test.app:id/title"
            class="android.widget.TextView"
            content-desc="" checkable="false" checked="false" clickable="false"
            enabled="true" focusable="false" focused="false" scrollable="false"
            long-clickable="false" password="false" selected="false"
            bounds="[48,48][300,120]" package="com.test.app"
            visible-to-user="true" important="false" />
      <node index="3" text="" resource-id="com.test.app:id/back_btn"
            class="android.widget.ImageButton"
            content-desc="Navigate up" checkable="false" checked="false"
            clickable="true" enabled="true" focusable="true" focused="false"
            scrollable="false" long-clickable="false" password="false"
            selected="false" bounds="[0,24][96,144]" package="com.test.app"
            visible-to-user="true" important="true" />
    </node>
    <node index="4" text="" resource-id="com.test.app:id/scroll"
          class="android.widget.ScrollView"
          content-desc="" checkable="false" checked="false" clickable="false"
          enabled="true" focusable="false" focused="false" scrollable="true"
          long-clickable="false" password="false" selected="false"
          bounds="[0,168][1080,1920]" package="com.test.app"
          visible-to-user="true" important="false">
      <node index="5" text="" resource-id=""
            class="android.widget.LinearLayout"
            content-desc="" checkable="false" checked="false" clickable="false"
            enabled="true" focusable="false" focused="false" scrollable="false"
            long-clickable="false" password="false" selected="false"
            bounds="[0,168][1080,1920]" package="com.test.app"
            visible-to-user="true" important="false">
        <node index="6" text="Dark mode" resource-id="com.test.app:id/dark_label"
              class="android.widget.TextView"
              content-desc="" checkable="false" checked="false" clickable="false"
              enabled="true" focusable="false" focused="false" scrollable="false"
              long-clickable="false" password="false" selected="false"
              bounds="[48,200][600,280]" package="com.test.app"
              visible-to-user="true" important="false" />
        <node index="7" text="" resource-id="com.test.app:id/dark_switch"
              class="android.widget.Switch"
              content-desc="Dark mode toggle" checkable="true" checked="false"
              clickable="true" enabled="true" focusable="true" focused="false"
              scrollable="false" long-clickable="false" password="false"
              selected="false" bounds="[900,200][1056,280]" package="com.test.app"
              visible-to-user="true" important="true" />
        <node index="8" text="Notifications" resource-id="com.test.app:id/notif_label"
              class="android.widget.TextView"
              content-desc="" checkable="false" checked="false" clickable="false"
              enabled="true" focusable="false" focused="false" scrollable="false"
              long-clickable="false" password="false" selected="false"
              bounds="[48,320][600,400]" package="com.test.app"
              visible-to-user="true" important="false" />
        <node index="9" text="" resource-id="com.test.app:id/notif_switch"
              class="android.widget.Switch"
              content-desc="Notifications toggle" checkable="true" checked="true"
              clickable="true" enabled="true" focusable="true" focused="false"
              scrollable="false" long-clickable="false" password="false"
              selected="false" bounds="[900,320][1056,400]" package="com.test.app"
              visible-to-user="true" important="true" />
        <node index="10" text="" resource-id="com.test.app:id/name_input"
              class="android.widget.EditText"
              content-desc="Display name" checkable="false" checked="false"
              clickable="true" enabled="true" focusable="true" focused="false"
              scrollable="false" long-clickable="false" password="false"
              selected="false" bounds="[48,440][1032,540]" package="com.test.app"
              visible-to-user="true" important="true" />
        <node index="11" text="" resource-id="com.test.app:id/profile_img"
              class="android.widget.ImageView"
              content-desc="Profile picture" checkable="false" checked="false"
              clickable="false" enabled="true" focusable="false" focused="false"
              scrollable="false" long-clickable="true" password="false"
              selected="false" bounds="[400,580][680,860]" package="com.test.app"
              visible-to-user="true" important="false" />
      </node>
    </node>
  </node>
</hierarchy>
"""
