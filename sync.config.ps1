$SyncSettings = [ordered]@{
    LocalDir        = 'D:\common'
    Branch          = 'main'
    RemoteName      = 'origin'

    # Blue computer has GitHub user email and SSH key configured.
    BlueRemoteUrl   = 'git@github.com:vikawq/yewllo_blue_sync.git'

    # Yellow computer downloads only. This requires the repository to be public,
    # or the yellow computer to have read-only GitHub access already configured.
    YellowRemoteUrl = 'https://github.com/vikawq/yewllo_blue_sync.git'
}
